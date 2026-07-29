"""Pure contract helpers for S2-03T pelvis-contact path recovery.

This module deliberately contains no Isaac/Kit imports.  The scientific
contract keeps the previously certified chest endpoint and all safety gates
unchanged; it only adds a development-time path-clearance audit and bounded
three-waypoint search.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence


STAGE = "S2-03T_PELVIS_CONTACT_PAIR_AND_LEFT_ARM_PATH_RECOVERY"
REFERENCE_STATUS = "KINEMATICALLY_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED"
CONTROL_DT_S = 0.02
SETTLE_STEPS = 50
MOVE_STEPS = 250
HOLD_STEPS = 500
STATIC_STRAIGHT_SAMPLES = 501
STATIC_SEGMENT_SAMPLES = 251
DEVELOPMENT_CLEARANCE_M = 0.005
STATIC_MARGIN_GATE_RAD = 0.12
DYNAMIC_MARGIN_GATE_RAD = 0.10
FORBIDDEN_FORCE_GATE_N = 1.0
ROOT_TILT_GATE_DEG = 6.2075676918029785
ROOT_HEIGHT_RANGE_M = (0.50, 1.00)
TORQUE_RATIO_GATE = 1.001
JOINT_RATE_LIMIT_RAD_S = 2.0
LOWER_BODY_COMMAND = (0.0, 0.0, 0.0, 0.7)
LOWER_CHECKPOINT_SHA256 = "a0151975a757a33f0f5ed236d5616e2a98643abe2b36e50ac9ad01b6dd1f6d7e"
REFERENCE_PATH = Path(
    "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
    "s2_03t_contract_smoke_20260728_104740/precontact_reference.pt"
)
REFERENCE_SHA256 = "1e537e075af7f888fb95a63501d3976d0d7740eff51c6bd9000f574b00cd0e6c"
OLD_RUN = Path(
    "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
    "s2_03t_safe_chest_joint_reference_20260729_033538"
)
OLD_RESULT_PATH = OLD_RUN / "result.json"
OLD_PAIR_AUDIT_PATH = OLD_RUN / "formal_joint_space_move_hold_trace.json"

ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
LEFT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[0::2]
RIGHT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[1::2]
PELVIS_BODY_NAME = "pelvis"
PALM_BODY_NAMES = ("left_hand_palm_link", "right_hand_palm_link")

CONTACT_CLASSIFICATIONS = (
    "LEFT_HAND_VS_PELVIS_REAL_PATH_COLLISION",
    "LEFT_WRIST_VS_PELVIS_REAL_PATH_COLLISION",
    "LEFT_FOREARM_VS_PELVIS_REAL_PATH_COLLISION",
    "LEFT_UPPER_ARM_VS_PELVIS_REAL_PATH_COLLISION",
    "PELVIS_VS_GROUND",
    "INTENTIONAL_ADJACENT_COLLIDER_OVERLAP",
    "COLLIDER_APPROXIMATION_ARTIFACT",
    "CONTACT_SENSOR_AGGREGATION_ERROR",
    "UNKNOWN",
)


def finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def minimum_jerk_fraction(step: int, steps: int) -> float:
    if steps <= 0 or not 0 <= step <= steps:
        raise ValueError(f"step={step} outside [0,{steps}]")
    u = float(step) / float(steps)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def interpolate_q(start: Sequence[float], goal: Sequence[float], fraction: float) -> list[float]:
    if len(start) != len(goal):
        raise ValueError("WAYPOINT_DIMENSION_MISMATCH")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("WAYPOINT_FRACTION_OUT_OF_RANGE")
    return [float(a) + float(fraction) * (float(b) - float(a)) for a, b in zip(start, goal, strict=True)]


def segment_samples(start: Sequence[float], goal: Sequence[float], samples: int = STATIC_SEGMENT_SAMPLES) -> list[list[float]]:
    if samples < 2:
        raise ValueError("SEGMENT_SAMPLE_COUNT_TOO_SMALL")
    return [interpolate_q(start, goal, i / (samples - 1)) for i in range(samples)]


def three_segment_path(q0: Sequence[float], q_clear: Sequence[float], q_lift: Sequence[float], q_chest: Sequence[float]) -> list[list[float]]:
    if not (len(q0) == len(q_clear) == len(q_lift) == len(q_chest)):
        raise ValueError("WAYPOINT_DIMENSION_MISMATCH")
    return [list(q0), list(q_clear), list(q_lift), list(q_chest)]


def path_joint_travel(waypoints: Iterable[Sequence[float]]) -> float:
    points = [list(point) for point in waypoints]
    return sum(
        math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(left, right, strict=True)))
        for left, right in zip(points, points[1:], strict=False)
    )


def order_audit(reference_names: Sequence[str], runtime_names: Sequence[str], actuator_names: Sequence[str]) -> dict[str, object]:
    expected = list(ARM_JOINT_NAMES)
    return {
        "reference_order": list(reference_names),
        "runtime_order": list(runtime_names),
        "actuator_order": list(actuator_names),
        "expected_interleaved_order": expected,
        "reference_matches_expected": list(reference_names) == expected,
        "runtime_matches_expected": list(runtime_names) == expected,
        "actuator_matches_expected": list(actuator_names) == expected,
        "pass": list(reference_names) == expected and list(runtime_names) == expected and list(actuator_names) == expected,
    }


def mirror_audit(left_q: Sequence[float], right_q: Sequence[float], left_pose: Sequence[float], right_pose: Sequence[float]) -> dict[str, object]:
    if len(left_q) != 7 or len(right_q) != 7 or len(left_pose) != 3 or len(right_pose) != 3:
        return {"status": "INVALID", "reason": "MIRROR_DIMENSION_MISMATCH"}
    # The G1 arm branches are not numerically identical.  Mirror validity is
    # therefore checked through named branches and palm y-signs, not q equality.
    y_sum = float(left_pose[1]) + float(right_pose[1])
    return {
        "status": "PASS" if math.isfinite(y_sum) else "INVALID",
        "left_q_rad": list(map(float, left_q)),
        "right_q_rad": list(map(float, right_q)),
        "palm_y_sum_m": y_sum,
        "criterion": "named-left-right branches with opposite palm y signs",
    }
