"""Pure contracts for the isolated S2-03T chest-prepose qualification.

This module deliberately contains no Isaac/Kit imports.  It records the
certified S2-02 geometry and the deterministic trajectory constants used by
the runtime runner, so the static contract can be checked without launching
the simulator.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


STAGE = "S2-03T_CHEST_PREPOSE_ONLY"
REFERENCE_PATH = Path(
    "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
    "s2_03t_contract_smoke_20260728_104740/precontact_reference.pt"
)
REFERENCE_SHA256 = "1e537e075af7f888fb95a63501d3976d0d7740eff51c6bd9000f574b00cd0e6c"
REFERENCE_PROVENANCE = "S2-03T_BOOTSTRAP_REPLAY_OF_S2-02_CERTIFIED_INDEX9"
S2_02_CONFIG = Path("configs/stage2/s2_02_precontact_audit.yaml")
S2_02_CONFIG_SHA256 = "7392cfdd98781a5699f3db4dc766ba0a6d6f5eaf5a40cf07e94d7f3518a8ae36"
S2_02_SOURCE_SYMBOL = "src/g1_access_push/stage2/s2_02_contract.py::object_local_targets"
S2_02_PELVIS_CONVERSION_SYMBOL = (
    "scripts/stage2_isaac/run_s2_02_precontact_audit.py::target_pose_in_pelvis"
)
S2_02_FORMAL_RUN = (
    "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
    "s2_02_formal_20260728_063446"
)
S2_02_SOURCE_COMMIT = "dea1104f26b0f8fcdb81d8cd225f6880d5075ca7"
S2_02_RESULT_STATUS = "PASS"
S2_02_CANDIDATE_INDEX = 9
S2_02_CANDIDATE = {
    "contact_height_m": 0.62,
    "tangential_separation_m": 0.30,
    "base_to_box_center_distance_m": 1.06,
    "precontact_gap_m": 0.06,
    "palm_collision_support_offset_m": 0.44023889869451527,
}
S2_02_DESIRED_PALM_QUATERNION_OBJECT_WXYZ = (
    0.7071067811865476,
    0.0,
    0.7071067811865476,
    0.0,
)
S2_02_PALM_LOCAL_NORMAL_AXIS = (0.0, 0.0, 1.0)

# This is the exact interleaved order resolved by the runtime articulation.
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
PALM_BODY_NAMES = ("left_hand_palm_link", "right_hand_palm_link")
PALM_FRAME_NAMES = ("left_hand_palm", "right_hand_palm")

CONTROL_DT_S = 0.02
SETTLE_STEPS = 100
MOVE_STEPS = 150
HOLD_STEPS = 500
HOLD_SECONDS = HOLD_STEPS * CONTROL_DT_S
MAX_POSITION_COMMAND_M = 0.005
MAX_ORIENTATION_COMMAND_RAD = 0.03
POSITION_P95_GATE_M = 0.03
POSITION_MAX_GATE_M = 0.06
ORIENTATION_P95_GATE_DEG = 5.0
ORIENTATION_MAX_GATE_DEG = 8.0
JOINT_MARGIN_GATE_RAD = 0.10
TORQUE_RATIO_GATE = 1.001
ROOT_HEIGHT_RANGE_M = (0.50, 1.00)
ROOT_TILT_GATE_DEG = 6.2075676918029785
FORBIDDEN_FORCE_GATE_N = 1.0


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one evidence/config file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def minimum_jerk_fraction(step: int, steps: int) -> float:
    """The monotonic 5th-order minimum-jerk interpolation fraction."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 <= step <= steps:
        raise ValueError(f"step must be in [0,{steps}], got {step}")
    u = float(step) / float(steps)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def certified_object_local_targets() -> list[list[float]]:
    """Return the immutable S2-02 candidate-9 palm origins in object frame."""

    rear_face_x = -0.6
    candidate = S2_02_CANDIDATE
    x = (
        rear_face_x
        - float(candidate["precontact_gap_m"])
        - float(candidate["palm_collision_support_offset_m"])
    )
    z = -0.6 + float(candidate["contact_height_m"])
    separation = float(candidate["tangential_separation_m"])
    return [[x, separation / 2.0, z], [x, -separation / 2.0, z]]


def validate_reference_metadata(reference: dict[str, Any]) -> list[str]:
    """Return contract violations for the frozen runtime reference payload."""

    failures: list[str] = []
    if reference.get("schema_version") != 1:
        failures.append("REFERENCE_SCHEMA_VERSION")
    if list(reference.get("arm_joint_names", ())) != list(ARM_JOINT_NAMES):
        failures.append("REFERENCE_ARM_JOINT_ORDER")
    if "arm_ik_target" not in reference:
        failures.append("REFERENCE_ARM_IK_TARGET_MISSING")
    if "robot_root_state_relative" not in reference:
        failures.append("REFERENCE_ROOT_STATE_MISSING")
    if "box_root_state_relative" not in reference:
        failures.append("REFERENCE_OBJECT_STATE_MISSING")
    return failures


def finite_number(value: object) -> bool:
    """Recursively check JSON-compatible numeric values for finiteness."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_number(item) for item in value)
    return True
