"""Geometry helpers for an elongated rectangular box."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from g1_access_push.task.coordinate_frames import SE2Pose, transform_points

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BoxGeometry:
    """Rigid rectangular box dimensions in metres."""

    length: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = np.asarray([self.length, self.width, self.height], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"Box dimensions must be finite and positive, got {values!r}")

    @property
    def corner_radius(self) -> float:
        return 0.5 * math.hypot(self.length, self.width)


def corners_local(box: BoxGeometry) -> FloatArray:
    """Return the four planar box corners in counter-clockwise order."""
    half_length = 0.5 * box.length
    half_width = 0.5 * box.width
    return np.array(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )


def corners_world(box: BoxGeometry, box_pose_world: SE2Pose) -> FloatArray:
    """Return box corners expressed in the world frame."""
    return transform_points(box_pose_world, corners_local(box))


def projected_width_perpendicular_to_door(box: BoxGeometry, alpha_rad: float) -> float:
    """Width projected onto the door-width axis.

    ``alpha_rad`` is the box long-axis yaw minus the door forward-axis yaw.
    """
    return box.length * abs(math.sin(alpha_rad)) + box.width * abs(math.cos(alpha_rad))


def half_extent_along_door(box: BoxGeometry, alpha_rad: float) -> float:
    """Half box extent projected onto the door forward axis."""
    return 0.5 * (
        box.length * abs(math.cos(alpha_rad)) + box.width * abs(math.sin(alpha_rad))
    )


def planar_wrench_from_forces(
    contact_points_xy: FloatArray,
    forces_xy: FloatArray,
) -> FloatArray:
    """Return ``[Fx, Fy, tau_z]`` from planar contacts and forces.

    Contact points are relative to the box centre. Positive ``tau_z`` follows the
    right-hand rule, i.e. counter-clockwise yaw about +z.
    """
    points = np.asarray(contact_points_xy, dtype=np.float64)
    forces = np.asarray(forces_xy, dtype=np.float64)
    if points.shape != forces.shape or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "contact_points_xy and forces_xy must both have shape (N, 2); "
            f"got {points.shape} and {forces.shape}"
        )
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(forces)):
        raise ValueError("Contact points and forces must be finite")

    force_sum = forces.sum(axis=0)
    torque_z = np.sum(points[:, 0] * forces[:, 1] - points[:, 1] * forces[:, 0])
    return np.array([force_sum[0], force_sum[1], torque_z], dtype=np.float64)
