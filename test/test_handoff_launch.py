"""Validate the standalone launch and real bag-metadata reader without Ouster hardware."""
import importlib.util
import json
from pathlib import Path

import pytest
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
from std_msgs.msg import String

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('handoff', PACKAGE / 'launch/dlio_ouster.launch.py')
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def test_points_mode_requires_only_dliio():
    actions = handoff.build_actions(dict(handoff.DEFAULTS), PACKAGE)
    assert len(actions) == 1  # one component container; no bag, RViz, or driver helper
    actions_without_map = handoff.build_actions(dict(handoff.DEFAULTS, map='false'), PACKAGE)
    assert len(actions_without_map) == 1


def test_persistent_mapper_is_a_separate_process(tmp_path):
    from launch_ros.actions import Node
    actions = handoff.build_actions(dict(handoff.DEFAULTS, mapper='persistent', run_dir=str(tmp_path)), PACKAGE)
    assert len(actions) == 2
    assert isinstance(actions[1], Node)
    assert len(handoff.build_actions(dict(handoff.DEFAULTS, mapper='persistent', map='false'), PACKAGE)) == 1


@pytest.mark.parametrize('key,value', [('mode', 'wrong'), ('profile', 'wrong'), ('rate', '0'),
                                    ('rate', 'nan'), ('rate', 'inf'), ('use_sim_time', 'maybe'),
                                    ('mapper', 'wrong'), ('mapper', '')])
def test_invalid_configuration_fails_before_starting_nodes(key, value):
    with pytest.raises(ValueError):
        handoff.build_actions(dict(handoff.DEFAULTS, **{key: value}), PACKAGE)


def test_packet_replay_requires_metadata_or_bag():
    with pytest.raises(ValueError, match='requires bag'):
        handoff.build_actions(dict(handoff.DEFAULTS, mode='packets'), PACKAGE)


def test_replay_cannot_use_wall_time(tmp_path):
    with pytest.raises(ValueError, match='requires use_sim_time'):
        handoff.build_actions(dict(handoff.DEFAULTS, bag=str(tmp_path), use_sim_time='false'), PACKAGE)


def write_metadata_bag(path, topic, data):
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(path), storage_id='sqlite3'), ConverterOptions('', ''))
    fields = dict(name=topic, type='std_msgs/msg/String', serialization_format='cdr')
    try:
        metadata = TopicMetadata(id=0, **fields)
    except TypeError:  # Humble predates the explicit topic ID argument.
        metadata = TopicMetadata(**fields)
    writer.create_topic(metadata)
    writer.write(topic, serialize_message(String(data=data)), 1000000000)
    del writer


def test_metadata_extraction_reads_real_bag_with_spaces(tmp_path):
    bag = tmp_path / 'bag with spaces'
    expected = {'sensor_info': {'prod_line': 'OS-1-64'}, 'lidar_mode': '1024x10'}
    write_metadata_bag(bag, '/ouster/metadata', json.dumps(expected))
    destination = tmp_path / 'run with spaces' / 'metadata.json'
    assert handoff.extract_metadata(bag, destination) == destination
    assert json.loads(destination.read_text()) == expected


def test_missing_metadata_is_actionable(tmp_path):
    bag = tmp_path / 'no-metadata'
    write_metadata_bag(bag, '/something_else', '{}')
    with pytest.raises(ValueError, match='metadata:='):
        handoff.extract_metadata(bag, tmp_path / 'metadata.json')


def test_corrupt_metadata_is_rejected(tmp_path):
    bag = tmp_path / 'corrupt'
    write_metadata_bag(bag, '/ouster/metadata', 'not json')
    with pytest.raises(ValueError):
        handoff.extract_metadata(bag, tmp_path / 'metadata.json')
