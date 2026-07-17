"""Small, dependency-light coordinate-frame utilities for Stage 0.

Conventions
-----------
- Right-handed world frame.
- +z points upward.
- Positive yaw is counter-clockwise about +z.
- ``compose(a, b)`` returns the pose obtained by applying ``a`` and then ``b``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to the half-open interval [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def rotation_matrix(yaw_rad: float) -> FloatArray:
    """Return the 2-D rotation matrix for ``yaw_rad``."""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SE2Pose:
    """Planar rigid-body pose ``(x, y, yaw)`` in SI units."""

    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        values = np.asarray([self.x, self.y, self.yaw], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"SE2Pose must be finite, got {values!r}")

    @property
    def translation(self) -> FloatArray:
        return np.array([self.x, self.y], dtype=np.float64)

    def as_matrix(self) -> FloatArray:
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2] = rotation_matrix(self.yaw)
        matrix[:2, 2] = self.translation
        return matrix


def compose(a: SE2Pose, b: SE2Pose) -> SE2Pose:
    """Compose poses: ``T_W_C = T_W_A @ T_A_C``."""
    translation = a.translation + rotation_matrix(a.yaw) @ b.translation
    return SE2Pose(
        x=float(translation[0]),
        y=float(translation[1]),
        yaw=wrap_to_pi(a.yaw + b.yaw),
    )


def inverse(pose: SE2Pose) -> SE2Pose:
    """Return the inverse planar transform."""
    rotation_inv = rotation_matrix(pose.yaw).T
    translation_inv = -(rotation_inv @ pose.translation)
    return SE2Pose(
        x=float(translation_inv[0]),
        y=float(translation_inv[1]),
        yaw=wrap_to_pi(-pose.yaw),
    )


def relative_pose(reference: SE2Pose, target: SE2Pose) -> SE2Pose:
    """Express ``target`` in ``reference`` coordinates."""
    return compose(inverse(reference), target)


def transform_points(pose: SE2Pose, points_xy: FloatArray) -> FloatArray:
    """Transform one point ``(2,)`` or an array of points ``(..., 2)``."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape == (2,):
        return rotation_matrix(pose.yaw) @ points + pose.translation
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError(f"Expected shape (2,) or (..., 2), got {points.shape}")
    return points @ rotation_matrix(pose.yaw).T + pose.translation
