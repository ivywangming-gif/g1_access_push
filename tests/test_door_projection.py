import math

import numpy as np

from g1_access_push.geometry.box_polygon import (
    BoxGeometry,
    projected_width_perpendicular_to_door,
)
from g1_access_push.geometry.doorway import (
    DoorGeometry,
    DoorRegion,
    classify_box_region,
    lateral_clearance,
    object_clears_exit,
)
from g1_access_push.task.coordinate_frames import SE2Pose


def test_projection_changes_from_width_to_length_at_ninety_degrees() -> None:
    box = BoxGeometry(length=1.2, width=0.4, height=0.5)
    assert math.isclose(projected_width_perpendicular_to_door(box, 0.0), 0.4)
    assert math.isclose(projected_width_perpendicular_to_door(box, math.pi / 2.0), 1.2)
    assert math.isclose(projected_width_perpendicular_to_door(box, math.pi), 0.4)


def test_lateral_clearance_accounts_for_offset_and_margin() -> None:
    box = BoxGeometry(length=1.2, width=0.4, height=0.5)
    door = DoorGeometry(pose_world=SE2Pose(0.0, 0.0, 0.0), width=1.0, thickness=0.2)
    centered = lateral_clearance(box, SE2Pose(-1.0, 0.0, 0.0), door, margin=0.05)
    offset = lateral_clearance(box, SE2Pose(-1.0, 0.1, 0.0), door, margin=0.05)
    assert math.isclose(centered, 0.25, abs_tol=1e-12)
    assert math.isclose(offset, 0.15, abs_tol=1e-12)


def test_region_and_full_exit_check() -> None:
    box = BoxGeometry(length=1.2, width=0.4, height=0.5)
    door = DoorGeometry(pose_world=SE2Pose(0.0, 0.0, 0.0), width=1.0, thickness=0.2)

    assert classify_box_region(box, SE2Pose(-1.0, 0.0, 0.0), door) == DoorRegion.PRE_DOOR
    assert classify_box_region(box, SE2Pose(0.0, 0.0, 0.0), door) == DoorRegion.THROAT
    assert classify_box_region(box, SE2Pose(1.0, 0.0, 0.0), door) == DoorRegion.POST_DOOR

    assert not object_clears_exit(box, SE2Pose(0.7, 0.0, 0.0), door, exit_margin=0.05)
    assert object_clears_exit(box, SE2Pose(0.8, 0.0, 0.0), door, exit_margin=0.05)


def test_world_rotation_of_door_does_not_change_local_geometry() -> None:
    box = BoxGeometry(length=1.2, width=0.4, height=0.5)
    door = DoorGeometry(
        pose_world=SE2Pose(2.0, -1.0, math.pi / 2.0),
        width=1.0,
        thickness=0.2,
    )
    # Box centre is one metre before the door along -x_D, which is world -y here.
    box_pose = SE2Pose(2.0, -2.0, math.pi / 2.0)
    assert np.isclose(lateral_clearance(box, box_pose, door), 0.3)
