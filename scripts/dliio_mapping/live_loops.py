"""Live revisit worker with a read-only SQLite snapshot and transactional ROS commits.

The storage worker remains the only writer and map/TF owner. Registration runs
in this separate process, using a consistent committed database view without
copying multi-gigabyte clouds or blocking sensor callbacks.
"""
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from .core import APPLICATION_ID, READ_VERSIONS, Limits, decode_points
from .graph import GraphStore, MAX_GRAPH_POSES, MAX_LOOPS, encoded
from .retrieval import anchor_ids, propose, settings


class LoopReadView(GraphStore):
    """Freeze small graph records; fetch immutable clouds without pinning WAL.

    Holding a disk read transaction during registration prevents checkpoints
    for minutes. Copy metadata under a short transaction instead. Original
    observation blobs are append-only and each lazy read must match the frozen
    count/checksum. Optimized poses and mutable submaps are never read lazily.
    """
    def __init__(self, path, limits=None):
        self.path, self.limits, self.read_only = Path(path).resolve(), limits or Limits(), True
        self.source = sqlite3.connect(self.path.as_uri()+'?mode=ro', uri=True, isolation_level=None)
        self.db = sqlite3.connect(':memory:')
        try:
            self.source.execute('PRAGMA query_only=ON')
            self.source.execute('BEGIN')
            if (self.source.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or
                    self.source.execute('PRAGMA user_version').fetchone()[0] not in READ_VERSIONS):
                raise ValueError('Not a DLIO mapping archive')
            self.meta = json.loads(self.source.execute('SELECT json FROM metadata WHERE id=1').fetchone()[0])
            count, first, last = self.source.execute('SELECT count(*),min(id),max(id) FROM keyframes').fetchone()
            if (not 2 <= count <= MAX_GRAPH_POSES or count != self.meta['keyframes'] or
                    first != 0 or last != count-1):
                raise ValueError('Archive observation IDs disagree with the committed metadata')
            if self.source.execute('SELECT count(*) FROM observations').fetchone()[0] != count:
                raise ValueError('Archive observation quality is incomplete')
            if self.source.execute('SELECT count(*) FROM graph_loops').fetchone()[0] > MAX_LOOPS:
                raise ValueError('Graph loop ledger exceeds capacity')
            for table, columns, size_column, maximum in (
                    ('metadata', 'id,json', 'json', 1048576),
                    ('keyframes', 'id,stamp_ns,pose,count,crc', 'pose', 128),
                    ('observations', 'id,json', 'json', 16384),
                    ('graph_state', 'id,configuration,pose_revision,covered_observations', 'configuration', 16384),
                    ('graph_loops', 'id,from_id,to_id,json,added_revision,removed_revision', 'json', 16384)):
                if self.source.execute(f'SELECT 1 FROM {table} WHERE length({size_column})>? LIMIT 1',
                                       (maximum,)).fetchone():
                    raise ValueError('Oversized loop input record')
                self.db.execute(f'CREATE TABLE {table} ({columns})')
                rows = self.source.execute(f'SELECT {columns} FROM {table} ORDER BY id').fetchall()
                placeholders = ','.join('?' for _ in columns.split(','))
                self.db.executemany(f'INSERT INTO {table} VALUES ({placeholders})', rows)
            self.db.commit()
            self.db.execute('CREATE UNIQUE INDEX keyframe_id ON keyframes(id)')
            self.db.execute('PRAGMA query_only=ON')
            self.source.execute('ROLLBACK')
        except BaseException:
            self.source.close()
            self.db.close()
            raise

    def _graph_cloud(self, identifier):
        frozen = self.db.execute('SELECT count,crc FROM keyframes WHERE id=?', (identifier,)).fetchone()
        row = self.source.execute('SELECT count,crc,points FROM keyframes WHERE id=?', (identifier,)).fetchone()
        if frozen is None or row is None or row[:2] != frozen:
            raise ValueError('Original observation changed after loop snapshot')
        return decode_points(row[2], row[0], row[1], self.limits.max_input_points)

    def close(self):
        self.source.close()
        self.db.close()


def main():
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from direct_lidar_inertial_odometry.srv import UpdatePoseGraph

    class LoopNode(Node):
        def __init__(self):
            super().__init__('dlio_loop_closure')
            filename = str(self.declare_parameter('loop_closure/configuration_file', '').value)
            self.config = settings(json.loads(Path(filename).expanduser().read_text()))
            self.output = Path(str(self.declare_parameter('loop_closure/output_directory', 'dliio_run/loops').value)).expanduser().resolve()
            self.output.mkdir(parents=True, exist_ok=True)
            self.session_directory = self.output/uuid.uuid4().hex
            self.session_directory.mkdir()
            (self.session_directory/'configuration.json').write_text(encoded(self.config)+'\n')
            self.client = self.create_client(UpdatePoseGraph, 'dlio/mapping/update_pose_graph')
            self.diagnostics = self.create_publisher(DiagnosticArray, 'diagnostics', 10)
            self.lock = threading.Lock()
            self.state = {}
            self.status = 'waiting for mapping observations'
            self.error = ''
            self.checked_count, self.checked_revision = 0, -1
            self.idle = False
            self.pending_anchors = 0
            self.stopping = threading.Event()
            self.subscription = self.create_subscription(DiagnosticArray, 'dlio/mapping/diagnostics', self.mapping, 10)
            self.timer = self.create_timer(1., self.diagnose, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.worker = threading.Thread(target=self.work, name='dliio-loop-registration', daemon=True)
            self.worker.start()
            self.get_logger().info('Automatic loop retrieval ready; original observations and held geometry are required')

        def mapping(self, message):
            for status in message.status:
                if status.name == 'DLIO Mapping':
                    with self.lock:
                        self.state = {item.key: item.value for item in status.values}

        def diagnose(self):
            with self.lock:
                values = dict(self.state, loop_status=self.status, loop_error=self.error,
                    loop_idle=self.idle, loop_checked_observations=self.checked_count,
                    loop_checked_pose_revision=self.checked_revision,
                    loop_pending_anchors=self.pending_anchors)
            status = DiagnosticStatus(name='DLIO Loop Closure', hardware_id='dliio',
                level=DiagnosticStatus.WARN if self.error else DiagnosticStatus.OK,
                message=self.status, values=[KeyValue(key=k, value=str(v)) for k, v in sorted(values.items())])
            message = DiagnosticArray()
            message.header.stamp = self.get_clock().now().to_msg()
            message.status = [status]
            self.diagnostics.publish(message)

        def call(self, payload):
            receipt = self.session_directory/f'request-{payload["request_id"]}.json'
            receipt.write_text(encoded(payload)+'\n')
            # The mapper may complete after its service response times out.
            # Retry the identical durable ID/body, never a fresh mutation.
            for attempt in range(3):
                response = self.call_once(payload)
                if response.success or not response.message.startswith('Request timed out;'):
                    receipt.with_suffix('.response.json').write_text(encoded(dict(
                        success=response.success, message=response.message,
                        pose_revision=response.pose_revision, report_json=response.report_json))+'\n')
                    if not response.success:
                        raise RuntimeError('Graph request rejected: '+response.message)
                    return response
            raise RuntimeError(f'Graph request {payload["request_id"]} remains unresolved; its exact payload is in {receipt}')

        def call_once(self, payload):
            deadline = time.monotonic()+300.
            while not self.stopping.is_set() and not self.client.wait_for_service(timeout_sec=1.):
                if time.monotonic() > deadline:
                    raise RuntimeError('Pose graph service unavailable')
            future = self.client.call_async(UpdatePoseGraph.Request(request_json=encoded(payload)))
            while not future.done() and not self.stopping.wait(.05):
                if time.monotonic() > deadline:
                    raise RuntimeError('Pose graph request is still pending; inspect the mapper before retrying')
            if not future.done():
                raise RuntimeError('Loop closure stopping')
            response = future.result()
            return response

        def work(self):
            session, completed_input, attempt = None, None, 0
            checked_sources = set()
            while not self.stopping.wait(1.):
                with self.lock:
                    state = dict(self.state)
                if not state.get('archive') or int(state.get('keyframes', 0)) < 4:
                    continue
                if session is not None and state.get('session_id') != session:
                    with self.lock:
                        self.status, self.error = 'stopped after archive session changed', 'Restart loop closure for the new archive'
                    return
                if state.get('read_only') == 'True':
                    continue
                count = int(state['keyframes'])
                committed_input = count, int(state.get('pose_revision', 0))
                if committed_input == completed_input:
                    continue
                attempt += 1
                report = None
                try:
                    view = LoopReadView(state['archive'])
                    try:
                        session = view.meta['session_id']
                        with self.lock:
                            self.status, self.error = 'checking revisit candidates', ''
                            self.idle = False
                        pending = [i for i in anchor_ids(view, self.config) if i not in checked_sources]
                        # Fresh revisits must not wait behind minutes of old
                        # rejected corridor matches. Backfill older anchors as
                        # the frontier is checked, including after sensor EOF.
                        report, _ = propose(view, self.config, sources=pending[-2:])
                        graph_stats = view.graph_stats()
                        initialized = graph_stats['graph_initialized']
                        at_capacity = graph_stats['active_loops']+len(report['loops']) >= self.config['retrieval']['max_loops']
                        revision = view.meta['pose_revision']
                        checked_count = view.meta['keyframes']
                    finally:
                        view.close()
                    if self.stopping.is_set():
                        return
                    path = self.session_directory/f'attempt-{attempt:06d}.json'
                    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                    if report['loops']:
                        def request(action, **extra):
                            return dict(session_id=session, expected_revision=revision,
                                        request_id=str(uuid.uuid4()), action=action, **extra)
                        if not initialized:
                            result = self.call(request('initialize', configuration=self.config['graph']))
                            revision = result.pose_revision
                        result = self.call(request('add_loops', loops=report['loops']))
                        revision = result.pose_revision
                        report['committed_revision'] = result.pose_revision
                        report['commit_report'] = json.loads(result.report_json)
                        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                        self.get_logger().info(f'Committed {len(report["loops"])} loop(s), map revision {result.pose_revision}')
                    checked_sources.update(report['processed_source_ids'])
                    remaining = sum(i not in checked_sources for i in pending)
                    idle = remaining == 0 or at_capacity
                    if idle:
                        completed_input = checked_count, revision
                    with self.lock:
                        self.idle, self.pending_anchors = idle, remaining
                        if self.idle:
                            self.checked_count, self.checked_revision = checked_count, revision
                        self.status = ('loop capacity reached' if at_capacity else
                            'waiting for new revisit support' if idle else 'checking recent revisits and backfilling earlier anchors')
                except Exception as error:
                    with self.lock:
                        self.status, self.error = 'loop correction requires attention', str(error)
                        self.idle = False
                    self.get_logger().warning(str(error))
                    (self.session_directory/f'error-{attempt:06d}.json').write_text(json.dumps(
                        dict(error=str(error), state=state), indent=2)+'\n')
                    # Wait for a changed committed input instead of repeatedly
                    # retrying an unchanged rejection or hogging the worker.
                    completed_input = committed_input

        def close(self):
            self.stopping.set()
            self.worker.join(timeout=5.)

    rclpy.init()
    node, executor = None, MultiThreadedExecutor(num_threads=2)
    try:
        node = LoopNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
