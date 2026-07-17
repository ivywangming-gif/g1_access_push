import math

import numpy as np

from g1_access_push.geometry.box_polygon import planar_wrench_from_forces
from g1_access_push.task.coordinate_frames import (
    SE2Pose,
    compose,
    inverse,
    relative_pose,
    transform_points,
)


def test_transform_chain_matches_hand_calculation() -> None:
    world_object = SE2Pose(1.0, 2.0, math.pi / 2.0)
    object_contact = SE2Pose(0.5, 0.0, 0.0)
    world_contact = compose(world_object, object_contact)

    assert np.allclose([world_contact.x, world_contact.y], [1.0, 2.5], atol=1e-12)
    assert math.isclose(world_contact.yaw, math.pi / 2.0, abs_tol=1e-12)


def test_inverse_and_relative_pose_round_trip() -> None:
    pose = SE2Pose(0.7, -0.2, 0.4)
    identity = compose(pose, inverse(pose))
    assert np.allclose([identity.x, identity.y, identity.yaw], [0.0, 0.0, 0.0], atol=1e-12)

    reference = SE2Pose(-0.3, 0.8, -0.5)
    target = SE2Pose(1.1, -0.4, 0.9)
    reconstructed = compose(reference, relative_pose(reference, target))
    assert np.allclose(
        [reconstructed.x, reconstructed.y, reconstructed.yaw],
        [target.x, target.y, target.yaw],
        atol=1e-12,
    )


def test_transform_points_accepts_single_and_batch() -> None:
    pose = SE2Pose(1.0, 1.0, math.pi / 2.0)
    single = transform_points(pose, np.array([1.0, 0.0]))
    batch = transform_points(pose, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(single, [1.0, 2.0], atol=1e-12)
    assert np.allclose(batch, [[1.0, 2.0], [0.0, 1.0]], atol=1e-12)


def test_analytic_differential_push_yaw_sign() -> None:
    # Rear-face contacts: left contact has +y, right contact has -y.
    contacts = np.array([[-0.5, 0.2], [-0.5, -0.2]], dtype=np.float64)
    forces = np.array([[10.0, 0.0], [15.0, 0.0]], dtype=np.float64)
    wrench = planar_wrench_from_forces(contacts, forces)
    assert np.allclose(wrench[:2], [25.0, 0.0])
    assert wrench[2] > 0.0  # stronger right push -> positive CCW yaw in our convention
