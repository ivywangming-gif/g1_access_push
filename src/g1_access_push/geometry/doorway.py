"""Door-frame geometry and conservative Stage-0 checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from g1_access_push.geometry.box_polygon import (
    BoxGeometry,
    half_extent_along_door,
    projected_width_perpendicular_to_door,
)
from g1_access_push.task.coordinate_frames import SE2Pose, relative_pose, wrap_to_pi

FloatArray = NDArray[np.float64]


class DoorRegion(StrEnum):
    PRE_DOOR = "pre_door"
    THROAT = "throat"
    POST_DOOR = "post_door"


@dataclass(frozen=True, slots=True)
class DoorGeometry:
    """Door frame in the world.

    The door-frame origin is the centre of the throat. ``+x_D`` points through
    the doorway and ``+y_D`` spans the door width.
    """

    pose_world: SE2Pose
    width: float
    thickness: float

    def __post_init__(self) -> None:
        values = np.asarray([self.width, self.thickness], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"Door width/thickness must be positive, got {values!r}")

    @property
    def entry_x_d(self) -> float:
        return -0.5 * self.thickness

    @property
    def exit_x_d(self) -> float:
        return 0.5 * self.thickness


def point_world_to_door(point_world_xy: FloatArray, door: DoorGeometry) -> FloatArray:
    """Express a world point in the door frame."""
    point = np.asarray(point_world_xy, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"Expected one finite point of shape (2,), got {point!r}")
    point_pose = SE2Pose(float(point[0]), float(point[1]), 0.0)
    relative = relative_pose(door.pose_world, point_pose)
    return np.array([relative.x, relative.y], dtype=np.float64)


def box_pose_world_to_door(box_pose_world: SE2Pose, door: DoorGeometry) -> SE2Pose:
    """Express a box pose in the door frame."""
    return relative_pose(door.pose_world, box_pose_world)


def lateral_clearance(
    box: BoxGeometry,
    box_pose_world: SE2Pose,
    door: DoorGeometry,
    margin: float = 0.0,
) -> float:
    """Return signed one-side door-width clearance in metres.

    Negative means that even the instantaneous box projection does not fit.
    """
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    pose_d = box_pose_world_to_door(box_pose_world, door)
    alpha = wrap_to_pi(box_pose_world.yaw - door.pose_world.yaw)
    projected_width = projected_width_perpendicular_to_door(box, alpha)
    return 0.5 * door.width - abs(pose_d.y) - 0.5 * projected_width - margin


def object_clears_exit(
    box: BoxGeometry,
    box_pose_world: SE2Pose,
    door: DoorGeometry,
    exit_margin: float = 0.0,
) -> bool:
    """Check whether the complete box has crossed the door exit plane."""
    if exit_margin < 0.0:
        raise ValueError("exit_margin must be non-negative")
    pose_d = box_pose_world_to_door(box_pose_world, door)
    alpha = wrap_to_pi(box_pose_world.yaw - door.pose_world.yaw)
    rear_x_d = pose_d.x - half_extent_along_door(box, alpha)
    return rear_x_d >= door.exit_x_d + exit_margin


def classify_box_region(
    box: BoxGeometry,
    box_pose_world: SE2Pose,
    door: DoorGeometry,
) -> DoorRegion:
    """Classify the box as before, intersecting, or after the door throat."""
    pose_d = box_pose_world_to_door(box_pose_world, door)
    alpha = wrap_to_pi(box_pose_world.yaw - door.pose_world.yaw)
    half_extent = half_extent_along_door(box, alpha)
    front_x_d = pose_d.x + half_extent
    rear_x_d = pose_d.x - half_extent
    if front_x_d < door.entry_x_d:
        return DoorRegion.PRE_DOOR
    if rear_x_d > door.exit_x_d:
        return DoorRegion.POST_DOOR
    return DoorRegion.THROAT
