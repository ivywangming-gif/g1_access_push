"""Frozen Stage-0 task success and failure taxonomy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from g1_access_push.geometry.box_polygon import BoxGeometry
from g1_access_push.geometry.doorway import DoorGeometry, object_clears_exit, point_world_to_door
from g1_access_push.task.coordinate_frames import SE2Pose, wrap_to_pi


class FailureCode(StrEnum):
    NONE = "NONE"
    F1_NO_PLAN = "F1_NO_PLAN"
    F2_STANCE_UNREACHABLE = "F2_STANCE_UNREACHABLE"
    F3_REACH_OR_BODY_COLLISION = "F3_REACH_OR_BODY_COLLISION"
    F4_CONTACT_OR_RESPONSE = "F4_CONTACT_OR_RESPONSE"
    F5_FALL_OR_TORQUE = "F5_FALL_OR_TORQUE"
    F6_DOOR_COLLISION = "F6_DOOR_COLLISION"
    F7_DOOR_STUCK = "F7_DOOR_STUCK"
    F8_TIMEOUT = "F8_TIMEOUT"


# Primary cause priority is deliberately fixed. Safety events outrank downstream symptoms.
_PRIMARY_PRIORITY = (
    FailureCode.F5_FALL_OR_TORQUE,
    FailureCode.F6_DOOR_COLLISION,
    FailureCode.F7_DOOR_STUCK,
    FailureCode.F4_CONTACT_OR_RESPONSE,
    FailureCode.F3_REACH_OR_BODY_COLLISION,
    FailureCode.F2_STANCE_UNREACHABLE,
    FailureCode.F1_NO_PLAN,
    FailureCode.F8_TIMEOUT,
)


@dataclass(frozen=True, slots=True)
class GoalPose:
    pose_world: SE2Pose
    position_tolerance_m: float
    yaw_tolerance_rad: float

    def __post_init__(self) -> None:
        if self.position_tolerance_m < 0.0 or self.yaw_tolerance_rad < 0.0:
            raise ValueError("Goal tolerances must be non-negative")


@dataclass(frozen=True, slots=True)
class SuccessCriteria:
    goal: GoalPose
    object_exit_margin_m: float
    robot_rear_radius_m: float
    robot_exit_margin_m: float

    def __post_init__(self) -> None:
        values = (
            self.object_exit_margin_m,
            self.robot_rear_radius_m,
            self.robot_exit_margin_m,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("Success clearances must be non-negative")


@dataclass(frozen=True, slots=True)
class TaskState:
    object_pose_world: SE2Pose
    robot_base_pose_world: SE2Pose
    fall: bool = False
    collision: bool = False


@dataclass(frozen=True, slots=True)
class SuccessReport:
    success: bool
    object_clear_door: bool
    robot_clear_door: bool
    goal_position_ok: bool
    goal_yaw_ok: bool
    no_fall: bool
    no_collision: bool
    position_error_m: float
    yaw_error_rad: float


@dataclass(frozen=True, slots=True)
class FailureSignals:
    no_plan: bool = False
    stance_unreachable: bool = False
    continuous_reach_failure: bool = False
    body_collision: bool = False
    contact_lost: bool = False
    object_response_wrong: bool = False
    fall: bool = False
    torque_violation: bool = False
    door_collision: bool = False
    door_stuck_unrecoverable: bool = False
    timeout: bool = False


@dataclass(frozen=True, slots=True)
class FailureReport:
    primary: FailureCode
    all_codes: tuple[FailureCode, ...]


def evaluate_success(
    state: TaskState,
    box: BoxGeometry,
    door: DoorGeometry,
    criteria: SuccessCriteria,
) -> SuccessReport:
    """Evaluate the frozen Stage-0 success contract."""
    object_clear = object_clears_exit(
        box=box,
        box_pose_world=state.object_pose_world,
        door=door,
        exit_margin=criteria.object_exit_margin_m,
    )

    robot_xy_d = point_world_to_door(state.robot_base_pose_world.translation, door)
    robot_rear_x_d = float(robot_xy_d[0]) - criteria.robot_rear_radius_m
    robot_clear = robot_rear_x_d >= door.exit_x_d + criteria.robot_exit_margin_m

    dx = state.object_pose_world.x - criteria.goal.pose_world.x
    dy = state.object_pose_world.y - criteria.goal.pose_world.y
    position_error = math.hypot(dx, dy)
    yaw_error = abs(wrap_to_pi(state.object_pose_world.yaw - criteria.goal.pose_world.yaw))

    position_ok = position_error <= criteria.goal.position_tolerance_m
    yaw_ok = yaw_error <= criteria.goal.yaw_tolerance_rad
    no_fall = not state.fall
    no_collision = not state.collision

    success = all((object_clear, robot_clear, position_ok, yaw_ok, no_fall, no_collision))
    return SuccessReport(
        success=success,
        object_clear_door=object_clear,
        robot_clear_door=robot_clear,
        goal_position_ok=position_ok,
        goal_yaw_ok=yaw_ok,
        no_fall=no_fall,
        no_collision=no_collision,
        position_error_m=position_error,
        yaw_error_rad=yaw_error,
    )


def classify_failure(signals: FailureSignals) -> FailureReport:
    """Return all matching codes plus one deterministic primary code."""
    matches: list[FailureCode] = []
    if signals.no_plan:
        matches.append(FailureCode.F1_NO_PLAN)
    if signals.stance_unreachable:
        matches.append(FailureCode.F2_STANCE_UNREACHABLE)
    if signals.continuous_reach_failure or signals.body_collision:
        matches.append(FailureCode.F3_REACH_OR_BODY_COLLISION)
    if signals.contact_lost or signals.object_response_wrong:
        matches.append(FailureCode.F4_CONTACT_OR_RESPONSE)
    if signals.fall or signals.torque_violation:
        matches.append(FailureCode.F5_FALL_OR_TORQUE)
    if signals.door_collision:
        matches.append(FailureCode.F6_DOOR_COLLISION)
    if signals.door_stuck_unrecoverable:
        matches.append(FailureCode.F7_DOOR_STUCK)
    if signals.timeout:
        matches.append(FailureCode.F8_TIMEOUT)

    if not matches:
        return FailureReport(primary=FailureCode.NONE, all_codes=())

    primary = next(code for code in _PRIMARY_PRIORITY if code in matches)
    return FailureReport(primary=primary, all_codes=tuple(matches))
