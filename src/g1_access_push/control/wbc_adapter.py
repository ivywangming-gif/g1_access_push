"""Simulator-agnostic interface between the research stack and WBC-AGILE.

Stage 0 intentionally contains no Isaac Sim or AGILE imports. A later adapter will
translate these data structures to the exact action/observation API discovered in
WBC-AGILE v1.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from g1_access_push.task.coordinate_frames import SE2Pose

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Pose3D:
    """3-D pose using metres and a quaternion in ``wxyz`` order."""

    position_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        position = np.asarray(self.position_xyz, dtype=np.float64)
        quaternion = np.asarray(self.quaternion_wxyz, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Pose3D expects position (3,) and quaternion (4,)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ValueError("Pose3D values must be finite")
        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError(f"Quaternion must be unit length, got norm={norm:.8f}")


@dataclass(frozen=True, slots=True)
class WBCCommand:
    """Minimum high-level command contract.

    Hand targets are expressed in the robot base/pelvis frame ``{B}``.
    """

    base_vx_mps: float
    base_vy_mps: float
    base_yaw_rate_rps: float
    base_height_m: float
    left_hand_pose_in_base: Pose3D
    right_hand_pose_in_base: Pose3D
    waist_yaw_rad: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.base_vx_mps,
                self.base_vy_mps,
                self.base_yaw_rate_rps,
                self.base_height_m,
                self.waist_yaw_rad,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("WBCCommand scalar values must be finite")
        if self.base_height_m <= 0.0:
            raise ValueError("base_height_m must be positive")


@dataclass(frozen=True, slots=True)
class WBCObservation:
    timestamp_s: float
    base_pose_world: SE2Pose
    left_hand_pose_world: Pose3D
    right_hand_pose_world: Pose3D
    left_contact: bool
    right_contact: bool
    fallen: bool
    max_joint_torque_ratio: float

    def __post_init__(self) -> None:
        values = np.asarray([self.timestamp_s, self.max_joint_torque_ratio], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("WBCObservation scalar values must be finite")
        if self.timestamp_s < 0.0 or self.max_joint_torque_ratio < 0.0:
            raise ValueError("timestamp_s and max_joint_torque_ratio must be non-negative")


class WBCAdapter(ABC):
    """Abstract adapter; simulator-specific code must implement this contract."""

    @property
    @abstractmethod
    def wbc_id(self) -> str:
        """Stable identifier containing checkpoint, gains, rates and observation version."""

    @abstractmethod
    def reset(self, seed: int) -> WBCObservation:
        """Reset one episode and return the first synchronized observation."""

    @abstractmethod
    def apply(self, command: WBCCommand) -> None:
        """Apply one high-level command without reading hidden global task state."""

    @abstractmethod
    def observe(self) -> WBCObservation:
        """Return the latest synchronized observation."""

    @abstractmethod
    def stop(self) -> None:
        """Send a safe zero-motion/hold command appropriate for the concrete WBC."""
