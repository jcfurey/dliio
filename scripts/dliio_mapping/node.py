"""ROS mapping node; all geometry and database work runs on its own worker."""
from array import array
from collections import Counter
from concurrent.futures import Future, TimeoutError
import json
import math
from pathlib import Path
import queue
import resource
import sqlite3
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TransformStamped
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from direct_lidar_inertial_odometry.msg import MappingObservation
from direct_lidar_inertial_odometry.srv import MapArchive, ApplyPoseRevision, RestorePoseRevision, UpdatePoseGraph

from .core import FIELDS, Limits, Store, validate_leaf
from .input import Pairer, decode_cloud, decode_pose, decode_observation, pose_message, stamp_ns


def cloud_message(points, stamp, frame):
    message = PointCloud2()
    message.header.frame_id = frame
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1000000000)
    message.height, message.width = 1, len(points)
    message.point_step, message.row_step = 28, 28 * len(points)
    message.is_dense = bool(np.isfinite(points).all())
    message.fields = [PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                      for i, name in enumerate(FIELDS)]
    message.data = array('B', np.ascontiguousarray(points, dtype='<f4').tobytes())
    return message


class MappingNode(Node):
    def __init__(self, node_name='dlio_mapping_node', **kwargs):
        super().__init__(node_name, automatically_declare_parameters_from_overrides=True, **kwargs)
        def parameter(name, default):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            return self.get_parameter(name).value
        self.limits = Limits(**{name: parameter('mapping/' + name, getattr(Limits(), name))
                               for name in Limits.__dataclass_fields__})
        self.fusion_size = parameter('mapping/fusion_size', .02)
        validate_leaf(self.fusion_size if self.fusion_size != 0 else None)
        self.odom_frame = parameter('frames/odom', 'odom')
        self.base_frame = parameter('frames/baselink', 'base_link')
        self.map_frame = parameter('frames/map', 'map')
        if (not all(isinstance(frame, str) and frame and not frame.startswith('/')
                    for frame in (self.odom_frame, self.base_frame, self.map_frame)) or
                self.map_frame == self.odom_frame):
            raise ValueError('Use nonempty relative frame names and distinct map/odom frames')
        for sensor in ('imu', 'lidar'):
            parameter(f'extrinsics/baselink2{sensor}/R', [1., 0., 0., 0., 1., 0., 0., 0., 1.])
            parameter(f'extrinsics/baselink2{sensor}/t', [0., 0., 0.])
        calibration = {name: value.value for name, value in self.get_parameters_by_prefix('').items()
                       if name.startswith(('extrinsics/', 'imu/', 'frames/'))}
        self.metadata = dict(odom_frame=self.odom_frame, base_frame=self.base_frame,
                             map_frame=self.map_frame, calibration=calibration,
                             input_source=parameter('mapping/input_source', 'keyframes'),
                             observation_selection={name: value.value for name, value in
                                 self.get_parameters_by_prefix('').items() if name.startswith('map/observation/')},
                             map_to_odom=np.eye(4).tolist())
        self.load_path = parameter('mapping/load_path', '')
        self.storage_directory = Path(parameter('mapping/storage_directory', 'dliio_run/mapping')).expanduser().resolve()
        self.viewer = bool(self.load_path)
        self.transport = parameter('mapping/transport', 'paired')
        if self.transport not in ('paired', 'observation'):
            raise ValueError('mapping/transport must be paired or observation')
        self.publish_tf = parameter('mapping/publish_tf', True)
        self.max_bytes = parameter('mapping/max_cloud_bytes', 16777216)
        if type(self.max_bytes) is not int or not 1024 <= self.max_bytes <= 67108864:
            raise ValueError('mapping/max_cloud_bytes must be in [1024, 67108864]')
        capacity = parameter('mapping/worker_queue', 4)
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError('mapping/worker_queue must be in [1, 64]')
        rate = parameter('mapping/publish_rate', 1.)
        if not math.isfinite(rate) or not .01 <= rate <= 20:
            raise ValueError('mapping/publish_rate must be in [.01, 20] Hz')
        self.pairer = Pairer(parameter('mapping/pending_pairs', 8), parameter('mapping/pair_timeout', 2.))
        self.jobs = queue.Queue(maxsize=capacity)
        self.lock, self.input_lock = threading.Lock(), threading.Lock()
        self.stopping = threading.Event()
        self.counters = Counter()
        self.state = {}
        self.last_error = ''
        self.storage_failed = False
        self.cached_map = None
        self.cached_correction = np.eye(4)
        self.revision = 0
        self.published_revision = -1
        self.refresh_interval = 1. / rate
        self.input_group = MutuallyExclusiveCallbackGroup()
        self.publish_group = MutuallyExclusiveCallbackGroup()
        # The storage worker serializes these operations anyway. Admit one
        # service callback at a time so waiting clients cannot occupy every
        # executor thread and starve input or diagnostics.
        self.service_group = MutuallyExclusiveCallbackGroup()
        self.map_pub = self.create_publisher(PointCloud2, 'map',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.diagnostics = self.create_publisher(DiagnosticArray, 'dlio/mapping/diagnostics', 10)
        # One dynamic authority owns map -> odom. Corrections and cloud caches
        # are swapped together after a complete revision rebuild.
        self.tf = TransformBroadcaster(self) if self.publish_tf else None
        self.ready = Future()
        self.worker = threading.Thread(target=self._work, name='dliio-mapping-storage', daemon=True)
        self.worker.start()
        try:
            self.ready.result(timeout=30)
        except BaseException:
            self.stopping.set()
            self.worker.join(timeout=30)
            raise
        if self.transport == 'paired':
            self.cloud_sub = self.create_subscription(PointCloud2, 'keyframes',
                lambda msg: self._input('cloud', msg), QoSProfile(depth=self.pairer.capacity), callback_group=self.input_group)
            self.pose_sub = self.create_subscription(PoseStamped, 'keyframe_pose',
                lambda msg: self._input('pose', msg), QoSProfile(depth=self.pairer.capacity), callback_group=self.input_group)
        else:
            self.observation_sub = self.create_subscription(MappingObservation, 'observation', self._observation,
                QoSProfile(depth=self.pairer.capacity), callback_group=self.input_group)
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.pair_timer = self.create_timer(.1, self._expire, clock=self.steady_clock, callback_group=self.input_group)
        self.publish_timer = self.create_timer(1. / rate, self._publish, clock=self.steady_clock,
                                              callback_group=self.publish_group)
        self.tf_timer = self.create_timer(.05, self._publish_tf, clock=self.steady_clock, callback_group=self.publish_group)
        self.diag_timer = self.create_timer(1., self._diagnose, clock=self.steady_clock, callback_group=self.publish_group)
        self.archive_services = [self.create_service(MapArchive, 'dlio/mapping/' + operation,
            lambda req, res, operation=operation: self._service(operation, req, res), callback_group=self.service_group)
            for operation in ('save_map', 'load_map', 'export_pcd')]
        self.revision_services = [self.create_service(kind, 'dlio/mapping/' + operation,
            lambda req, res, operation=operation: self._revision_service(operation, req, res),
            callback_group=self.service_group) for operation, kind in
            (('apply_pose_revision', ApplyPoseRevision), ('restore_pose_revision', RestorePoseRevision))]
        self.graph_service = self.create_service(UpdatePoseGraph, 'dlio/mapping/update_pose_graph',
                                                 self._graph_service, callback_group=self.service_group)
        self.add_on_set_parameters_callback(self._on_parameters)
        self.get_logger().info(f'Mapping archive: {self.state["archive"]}; '
                               f'{"read-only viewer" if self.viewer else "recording paired mapping frames"}')

    def _on_parameters(self, parameters):
        immutable = any(p.name.startswith(('mapping/', 'map/observation/', 'frames/', 'extrinsics/', 'imu/'))
                        and p.value != self.get_parameter(p.name).value for p in parameters)
        return SetParametersResult(successful=not immutable,
            reason='Restart the mapper to change configuration or calibration' if immutable else '')

    def _increment(self, name, count=1):
        with self.lock:
            self.counters[name] += count

    def _input(self, kind, message):
        if self.viewer or self.storage_failed:
            self._increment('input_ignored')
            return
        try:
            if message.header.frame_id != self.odom_frame:
                raise ValueError('Keyframe cloud and pose must both be in the configured odom frame')
            stamp = stamp_ns(message.header)
            if kind == 'cloud' and (len(message.data) > self.max_bytes or
                    message.width * message.height > self.limits.max_input_points):
                raise ValueError('Incoming cloud exceeds mapping input limits')
            with self.input_lock:
                self._enqueue(self.pairer.push(kind, stamp, message))
        except ValueError as error:
            self._increment('invalid_input')
            with self.lock:
                self.last_error = str(error)

    def _expire(self):
        with self.input_lock:
            self._enqueue(self.pairer.drain())

    def _observation(self, message):
        if self.viewer or self.storage_failed:
            self._increment('input_ignored')
            return
        if (len(message.cloud.data) > self.max_bytes or
                message.cloud.width * message.cloud.height > self.limits.max_input_points):
            self._increment('invalid_input')
            return
        try:
            self.jobs.put_nowait(('observation', message, None))
            self._increment('observations_received')
        except queue.Full:
            self._increment('queue_dropped')

    def _enqueue(self, pairs):
        for pair in pairs:
            try:
                self.jobs.put_nowait(('frame', pair, None))
                with self.lock:
                    self.counters['queue_peak'] = max(self.counters['queue_peak'], self.jobs.qsize())
            except queue.Full:
                self._increment('queue_dropped')

    def _open(self, path):
        store = Store(path, self.limits)
        if (store.meta['odom_frame'] != self.odom_frame or store.meta['map_frame'] != self.map_frame or
                store.meta['base_frame'] != self.base_frame):
            store.close()
            raise ValueError('Archive map/odom frames differ from the configured frames')
        return store

    def _refresh(self, store):
        start = time.monotonic()
        points = store.snapshot(self.fusion_size or None)
        message = cloud_message(points, store.meta['last_stamp_ns'], self.map_frame)
        with self.lock:
            self.cached_map = message
            self.cached_correction = np.asarray(store.meta['map_to_odom']).copy()
            self.state = store.stats()
            self.state['cached_pose_revision'] = store.meta['pose_revision']
            self.state['cached_map_bytes'] = len(message.data)
            self.state['published_points'] = len(points)
            self.state['fusion_size'] = self.fusion_size
            elapsed = (time.monotonic() - start) * 1000
            self.state['refresh_last_ms'] = elapsed
            self.counters['refresh_max_ms'] = max(self.counters['refresh_max_ms'], elapsed)
            self.revision += 1

    def _work(self):
        store = None
        try:
            store = (self._open(self.load_path) if self.viewer else
                     Store(self.storage_directory / ('session-' + uuid.uuid4().hex + '.dliomap'),
                           self.limits, metadata=self.metadata))
            self._refresh(store)
            self.ready.set_result(True)
            dirty = False
            next_refresh = time.monotonic() + self.refresh_interval
            while not self.stopping.is_set() or not self.jobs.empty():
                if dirty and (time.monotonic() >= next_refresh or self.stopping.is_set()):
                    self._refresh(store)
                    dirty = False
                    next_refresh = time.monotonic() + self.refresh_interval
                try:
                    operation, data, future = self.jobs.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    if future is not None and not future.set_running_or_notify_cancel():
                        continue
                    if operation in ('frame', 'observation'):
                        if self.viewer or self.storage_failed:
                            self._increment('input_ignored')
                            continue
                        start = time.monotonic()
                        if operation == 'observation':
                            stamp, pose, points, provenance = decode_observation(data, self.odom_frame, self.base_frame,
                                self.limits.max_input_points, self.max_bytes)
                            store.ingest(stamp, pose, points, observation=provenance)
                        else:
                            stamp, pose, cloud = data
                            store.ingest(stamp, decode_pose(pose),
                                         decode_cloud(cloud, self.limits.max_input_points, self.max_bytes))
                        dirty = True
                        elapsed = (time.monotonic() - start) * 1000
                        with self.lock:
                            self.counters['processed'] += 1
                            self.state.update(store.stats())
                            self.state['worker_last_ms'] = elapsed
                            self.counters['worker_max_ms'] = max(self.counters['worker_max_ms'], elapsed)
                    elif operation == 'save_map':
                        store.save(data)
                    elif operation == 'export_pcd':
                        store.export_pcd(data, self.fusion_size or None)
                    elif operation == 'update_pose_graph':
                        report = store.update_graph(data)
                        dirty = dirty or report['status'] == 'accepted'
                    elif operation == 'apply_pose_revision':
                        if data.frame_id != self.map_frame or len(data.observation_ids) != len(data.poses):
                            raise ValueError('Correction frame or pose/ID array lengths are invalid')
                        poses = [(int(identifier), decode_pose(SimpleNamespace(pose=pose)))
                                 for identifier, pose in zip(data.observation_ids, data.poses)]
                        store.apply_revision(data.session_id, int(data.expected_revision), poses,
                                             request_id=data.request_id, reason=data.reason)
                        dirty = True
                    elif operation == 'restore_pose_revision':
                        store.restore_revision(data.session_id, int(data.expected_revision), int(data.target_revision),
                                               request_id=data.request_id)
                        dirty = True
                    elif operation == 'load_map':
                        if not self.viewer:
                            raise ValueError('Loading into a recording session is disabled; start a viewer with load_map:=...')
                        candidate = self._open(data)  # validation completes before replacing current state
                        try:
                            self._refresh(candidate)
                        except BaseException:
                            candidate.close()
                            raise
                        store.close()
                        store = candidate
                    if future is not None:
                        future.set_result(dict(store.stats(), graph_report=report) if operation == 'update_pose_graph'
                                          else store.stats())
                except Exception as error:
                    with self.lock:
                        self.last_error = str(error)
                        self.counters['processing_errors' if operation in ('frame', 'observation') else 'service_errors'] += 1
                        if operation in ('frame', 'observation') and isinstance(error, (OSError, sqlite3.Error)):
                            self.storage_failed = True
                    if future is not None:
                        future.set_exception(error)
                finally:
                    self.jobs.task_done()
        except BaseException as error:
            if not self.ready.done():
                self.ready.set_exception(error)
            else:
                with self.lock:
                    self.storage_failed = True
                    self.last_error = str(error)
        finally:
            if store is not None:
                store.close()

    def _service(self, operation, request, response):
        future = Future()
        try:
            if not Path(request.path).is_absolute():
                raise ValueError('Archive/export paths must be absolute')
            if not self.worker.is_alive():
                raise RuntimeError('Mapping worker is unavailable')
            self.jobs.put_nowait((operation, request.path, future))
            stats = future.result(timeout=60)
            response.success = True
            response.message = str(request.path)
            response.keyframes, response.submaps = stats['keyframes'], stats['submaps']
        except TimeoutError:
            cancelled = future.cancel()
            response.message = 'Request timed out; ' + ('cancelled before execution' if cancelled else 'operation may still finish')
        except Exception as error:
            response.message = str(error) or 'Mapping worker queue is full'
        return response

    def _publish(self):
        with self.lock:
            message, revision = self.cached_map, self.revision
        if message is not None and revision != self.published_revision and self.map_pub.get_subscription_count():
            self.map_pub.publish(message)
            self.published_revision = revision

    def _revision_service(self, operation, request, response):
        future = Future()
        try:
            if not self.worker.is_alive():
                raise RuntimeError('Mapping worker is unavailable')
            self.jobs.put_nowait((operation, request, future))
            stats = future.result(timeout=60)
            response.success = True
            response.pose_revision = stats['pose_revision']
            response.keyframes, response.submaps = stats['keyframes'], stats['submaps']
            response.message = 'Revision committed; map/TF publication follows cache refresh'
        except TimeoutError:
            cancelled = future.cancel()
            response.message = 'Request timed out; ' + ('cancelled before execution' if cancelled else
                'operation may still finish; retry the same request ID to determine its result')
        except Exception as error:
            response.message = str(error) or 'Mapping worker queue is full'
        return response

    def _graph_service(self, request, response):
        from .graph import MAX_REQUEST_BYTES, encoded
        future = Future()
        try:
            if len(request.request_json.encode('utf8')) > MAX_REQUEST_BYTES:
                raise ValueError('Graph request exceeds 1 MiB')
            data = json.loads(request.request_json)
            if not self.worker.is_alive():
                raise RuntimeError('Mapping worker is unavailable')
            self.jobs.put_nowait(('update_pose_graph', data, future))
            stats = future.result(timeout=60)
            report = stats['graph_report']
            response.success = report['status'] == 'accepted'
            response.pose_revision = stats['pose_revision']
            response.message = report['reason']
            response.report_json = encoded(report)
        except TimeoutError:
            cancelled = future.cancel()
            response.message = 'Request timed out; ' + ('cancelled before execution' if cancelled else
                'operation may still finish; retry the identical request ID to determine its result')
        except Exception as error:
            response.message = str(error) or 'Mapping worker queue is full'
        return response

    def _publish_tf(self):
        if self.tf is None:
            return
        stamp = self.get_clock().now()
        if stamp.nanoseconds <= 0:
            return
        with self.lock:
            correction = self.cached_correction.copy()
        pose = pose_message(correction)
        message = TransformStamped()
        message.header.stamp = stamp.to_msg()
        message.header.frame_id, message.child_frame_id = self.map_frame, self.odom_frame
        message.transform.translation.x = pose.position.x
        message.transform.translation.y = pose.position.y
        message.transform.translation.z = pose.position.z
        message.transform.rotation = pose.orientation
        self.tf.sendTransform(message)

    def _diagnose(self):
        with self.lock:
            values = dict(self.state, **self.counters, queue_depth=self.jobs.qsize())
            error, failed = self.last_error, self.storage_failed
        with self.input_lock:
            values.update(self.pairer.counts)
            values['pending_pairs'] = len(self.pairer.pending)
        values['rss_peak_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        drops = sum(values.get(name, 0) for name in
                    ('missing_cloud', 'missing_pose', 'queue_dropped', 'processing_errors', 'invalid_input',
                     'late_or_duplicate', 'duplicate_pending', 'source_sequence_gaps'))
        status = DiagnosticStatus(name='DLIO Mapping', hardware_id=values.get('session_id', ''))
        status.level = DiagnosticStatus.ERROR if failed else DiagnosticStatus.WARN if drops else DiagnosticStatus.OK
        status.message = ('Storage failed; recording stopped: ' + error) if failed else (
            error or ('Read-only map viewer' if self.viewer else 'Recording paired mapping frames'))
        status.values = [KeyValue(key=key, value=str(value)) for key, value in sorted(values.items())]
        report = DiagnosticArray()
        report.header.stamp = self.get_clock().now().to_msg()
        report.status = [status]
        self.diagnostics.publish(report)

    def close(self):
        self.stopping.set()
        self.worker.join()


def main():
    rclpy.init()
    node = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        node = MappingNode()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
