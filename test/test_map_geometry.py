"""Dense native geometry: independent transforms, scalar bits and bounded failures."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from _dliio_pose_graph import reconstruct_dense_submap
from dliio_mapping.core import Limits, Store, rigid_pose, transform


def scene():
    rng = np.random.default_rng(264)
    clouds = [rng.uniform(-20., 20., (n, 7)).astype('<f4') for n in (13, 100, 73)]
    for cloud in clouds:
        cloud[1, :3] = cloud[0, :3]  # coincident samples must remain distinct
        cloud.view('<u4')[::3, 4] = 0x7fc00013  # noncanonical NaN payload
        cloud[::5, 5] = -0.
    poses = []
    for angles, shift in [([.2, .4, -.3], [100., -10., 5.]),
                          ([.1, -.2, .6], [104., -8., 6.]),
                          ([-.3, .1, .8], [106., -7., 7.])]:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(angles).as_matrix()
        pose[:3, 3] = shift
        poses.append(pose)
    relative = [np.eye(4)] + [np.linalg.inv(poses[0]) @ p for p in poses[1:]]
    return clouds, poses, relative


def test_nonuniform_geometry_matches_independent_transform_and_preserves_scalar_bits():
    clouds, poses, relative = scene()
    originals = [cloud.tobytes() for cloud in clouds]
    result = reconstruct_dense_submap(clouds, poses, relative, 1000)
    expected = np.vstack([clouds[0]] + [transform(c, p) for c, p in zip(clouds[1:], relative[1:])])
    np.testing.assert_allclose(result[:, :3], expected[:, :3], atol=4.e-6, rtol=0)
    np.testing.assert_array_equal(result[:, 3:].view('<u4'), expected[:, 3:].view('<u4'))
    np.testing.assert_array_equal(result[:len(clouds[0])].view('<u4'), clouds[0].view('<u4'))
    assert [cloud.tobytes() for cloud in clouds] == originals
    saved = result.tobytes()
    for cloud in clouds:
        cloud[:] = np.nan
    assert result.tobytes() == saved


def test_readonly_archive_buffers_and_tolerance_boundary_poses():
    cloud = np.zeros((3, 7), '<f4')
    cloud[:, :3] = [1., 2., 3.]
    cloud = np.frombuffer(cloud.tobytes(), '<f4').reshape(-1, 7)
    poses = [np.eye(4), np.eye(4)]
    poses[0][0, 0] -= 4.e-7
    poses[1][0, 0] += 4.e-7
    for pose in poses:
        rigid_pose(pose)
    relative = [np.eye(4), np.linalg.inv(poses[0]) @ poses[1]]
    result = reconstruct_dense_submap([cloud, cloud], poses, relative, 6)
    np.testing.assert_allclose(result[3:], transform(cloud, relative[1]), atol=1.e-6)


@pytest.mark.parametrize('bad', [np.zeros((0, 7), '<f4'), np.zeros((10, 6), '<f4'),
    np.zeros((10, 7), '<f8'), np.zeros((10, 14), '<f4')[:, ::2],
    np.zeros((10, 7), '>f4'), np.full((10, 7), np.nan, '<f4'), np.full((10, 7), np.inf, '<f4')])
def test_invalid_arrays_fail(bad):
    with pytest.raises(ValueError):
        reconstruct_dense_submap([bad], [np.eye(4)], [np.eye(4)], 100)


@pytest.mark.parametrize('max_points', [0, 185, 1000001])
def test_capacity_fails_before_output(max_points):
    with pytest.raises(ValueError, match='bounded|capacity'):
        reconstruct_dense_submap(*scene(), max_points)


def test_invalid_pose_and_world_range_fail_without_changing_inputs():
    clouds, poses, relative = scene()
    originals = [c.tobytes() for c in clouds]
    for invalid in (np.eye(3), np.full((4, 4), np.nan), np.diag([-1., 1., 1., 1.])):
        with pytest.raises(ValueError):
            reconstruct_dense_submap(clouds, [invalid]+poses[1:], relative, 1000)
    for shift in (1.e16, 1.e300):
        invalid = np.eye(4)
        invalid[0, 3] = shift
        with pytest.raises(ValueError, match='coordinate range|float32'):
            reconstruct_dense_submap(clouds, [invalid]+poses[1:], relative, 1000)
    assert [c.tobytes() for c in clouds] == originals
    with pytest.raises(ValueError, match='bounded'):
        reconstruct_dense_submap([], [], [], 100)
    with pytest.raises(ValueError, match='bounded'):
        reconstruct_dense_submap(clouds, poses[:-1], relative, 1000)


def test_bad_backend_cannot_create_database(tmp_path):
    path = tmp_path/'never-created.dliomap'
    with pytest.raises(ValueError, match='reconstruction_backend'):
        Store(path, Limits(), metadata={}, reconstruction_backend='typo')
    assert not path.exists()
