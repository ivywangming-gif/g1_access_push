"""Pure contracts for the S2-03T safe chest joint-reference recovery.

This module intentionally has no Isaac/Kit dependency.  It contains the
frozen geometry, the old-run audit helpers, and the deterministic joint-space
trajectory constants used by the runtime qualification runner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


STAGE = "S2-03T_SAFE_CHEST_JOINT_REFERENCE_RECOVERY"
OLD_RUN = Path(
    "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
    "s2_03t_chest_prepose_20260729_022014"
)
OLD_TRACE_PATH = OLD_RUN / "formal_trace.json"
OLD_GEOMETRY_AUDIT_PATH = Path(
    "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
    "s2_02_formal_20260728_063446/runtime_geometry_audit.json"
)
REFERENCE_PATH = Path(
    "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
    "s2_03t_contract_smoke_20260728_104740/precontact_reference.pt"
)
REFERENCE_SHA256 = "1e537e075af7f888fb95a63501d3976d0d7740eff51c6bd9000f574b00cd0e6c"
LOWER_CHECKPOINT_SHA256 = "a0151975a757a33f0f5ed236d5616e2a98643abe2b36e50ac9ad01b6dd1f6d7e"
S2_02_FORMAL_RUN = (
    "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
    "s2_02_formal_20260728_063446"
)
S2_02_SOURCE_SYMBOL = "src/g1_access_push/stage2/s2_02_contract.py::object_local_targets"
S2_02_PELVIS_SYMBOL = "scripts/stage2_isaac/run_s2_02_precontact_audit.py::target_pose_in_pelvis"

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

# The object and orientation are frozen by the S2-02 candidate-9 audit.
S2_02_CANDIDATE = {
    "candidate_index": 9,
    "contact_height_m": 0.62,
    "tangential_separation_m": 0.30,
    "base_to_box_center_distance_m": 1.06,
    "precontact_gap_m": 0.06,
    "palm_collision_support_offset_m": 0.44023889869451527,
}
S2_02_PALM_QUATERNION_OBJECT_WXYZ = (
    0.7071067811865476,
    0.0,
    0.7071067811865476,
    0.0,
)
S2_02_PALM_LOCAL_NORMAL_AXIS = (0.0, 0.0, 1.0)

CONTROL_DT_S = 0.02
SETTLE_STEPS = 50
MOVE_STEPS = 250  # five seconds at the verified 50-Hz control rate
SHORT_HOLD_STEPS = 100
FORMAL_HOLD_STEPS = 500
FORMAL_HOLD_SECONDS = FORMAL_HOLD_STEPS * CONTROL_DT_S
JOINT_MARGIN_GATE_RAD = 0.10
STATIC_MARGIN_GATE_RAD = 0.12
TORQUE_RATIO_GATE = 1.001
ROOT_HEIGHT_RANGE_M = (0.50, 1.00)
ROOT_TILT_GATE_DEG = 6.2075676918029785
FORBIDDEN_FORCE_GATE_N = 1.0
JOINT_RATE_LIMIT_RAD_S = 2.0


def certified_object_local_targets() -> list[list[float]]:
    """Return the two certified palm origins in the object frame."""

    rear_face_x = -0.6
    x = (
        rear_face_x
        - S2_02_CANDIDATE["precontact_gap_m"]
        - S2_02_CANDIDATE["palm_collision_support_offset_m"]
    )
    z = -0.6 + S2_02_CANDIDATE["contact_height_m"]
    y = S2_02_CANDIDATE["tangential_separation_m"] / 2.0
    return [[x, y, z], [x, -y, z]]


def minimum_jerk_fraction(step: int, steps: int) -> float:
    if steps <= 0 or not 0 <= step <= steps:
        raise ValueError(f"step={step} outside [0,{steps}]")
    u = float(step) / float(steps)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def finite_number(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite_number(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_number(v) for v in value)
    return True


def _margins(q: Iterable[float], limits: list[list[float]]) -> list[float]:
    return [min(float(value) - lo, hi - float(value)) for value, (lo, hi) in zip(q, limits, strict=True)]


def audit_old_trace(trace: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    """Recompute the old target/actual margin history from the complete trace."""

    records = list(trace.get("records", []))
    names = list(geometry["arm_joint_names"])
    limits = [list(pair) for pair in geometry["arm_joint_limits_rad"]]
    if len(records) == 0 or len(names) != 14 or len(limits) != 14:
        raise ValueError("OLD_TRACE_OR_LIMITS_INCOMPLETE")

    history: list[dict[str, Any]] = []
    max_actual_step = 0.0
    max_target_step = 0.0
    max_actual_step_joint = None
    max_target_step_joint = None
    previous_actual: list[float] | None = None
    previous_target: list[float] | None = None
    for record in records:
        actual = [float(v) for v in record["actual_upper_body_joints_rad"]]
        target = [float(v) for v in record["desired_upper_body_joints_rad"]]
        actual_margin = _margins(actual, limits)
        target_margin = _margins(target, limits)
        actual_min_index = min(range(14), key=actual_margin.__getitem__)
        target_min_index = min(range(14), key=target_margin.__getitem__)
        if previous_actual is not None:
            deltas = [abs(a - b) for a, b in zip(actual, previous_actual, strict=True)]
            index = max(range(14), key=deltas.__getitem__)
            if deltas[index] > max_actual_step:
                max_actual_step = deltas[index]
                max_actual_step_joint = names[index]
        if previous_target is not None:
            deltas = [abs(a - b) for a, b in zip(target, previous_target, strict=True)]
            index = max(range(14), key=deltas.__getitem__)
            if deltas[index] > max_target_step:
                max_target_step = deltas[index]
                max_target_step_joint = names[index]
        history.append(
            {
                "frame": int(record["frame"]),
                "time_s": float(record["time_s"]),
                "phase": record["phase"],
                "actual_margin_rad": min(actual_margin),
                "target_margin_rad": min(target_margin),
                "actual_limiting_joint": names[actual_min_index],
                "target_limiting_joint": names[target_min_index],
                "actual_margin_by_joint_rad": dict(zip(names, actual_margin, strict=True)),
                "target_margin_by_joint_rad": dict(zip(names, target_margin, strict=True)),
            }
        )
        previous_actual = actual
        previous_target = target

    last = history[-1]
    pre_move = [item for item in history if item["phase"] != "FORMAL_MOVE"]
    first_target_below = next((item for item in history if item["target_margin_rad"] < JOINT_MARGIN_GATE_RAD), None)
    first_actual_below = next((item for item in history if item["actual_margin_rad"] < JOINT_MARGIN_GATE_RAD), None)
    requested = records[-1]["desired_upper_body_joints_rad"]
    final_target = records[-1].get("final_actuator_target_rad")
    clip_observable = final_target is not None and len(final_target) == 14
    return {
        "old_run": str(OLD_RUN),
        "trace_records": len(records),
        "joint_names": names,
        "joint_limits_rad": limits,
        "trigger_frame": int(records[-1]["frame"]),
        "trigger_time_s": float(records[-1]["time_s"]),
        "limiting_joint": last["target_limiting_joint"],
        "limiting_joint_index": names.index(last["target_limiting_joint"]),
        "lower_limit_rad": limits[names.index(last["target_limiting_joint"])][0],
        "upper_limit_rad": limits[names.index(last["target_limiting_joint"])][1],
        "actual_joint_position_rad": records[-1]["actual_upper_body_joints_rad"][names.index(last["target_limiting_joint"])],
        "requested_target_joint_position_rad": requested[names.index(last["target_limiting_joint"])],
        "actual_margin_rad": last["actual_margin_rad"],
        "target_margin_rad": last["target_margin_rad"],
        "target_margin_history": history,
        "pre_formal_move_min_target_margin_rad": min(item["target_margin_rad"] for item in pre_move),
        "pre_formal_move_target_margin_below_0p10": any(item["target_margin_rad"] < JOINT_MARGIN_GATE_RAD for item in pre_move),
        "first_target_below_0p10": first_target_below,
        "first_actual_below_0p10": first_actual_below,
        "clip_before_rad": "METRIC_MISSING",
        "clip_after_rad": final_target if clip_observable else "METRIC_MISSING",
        "clip_evidence": "TRACE_HAS_NO_PRE_CLIP_FIELD; final_actuator_target_equals_requested_target",
        "max_actual_delta_q_rad": max_actual_step,
        "max_actual_delta_q_joint": max_actual_step_joint,
        "max_target_delta_q_rad": max_target_step,
        "max_target_delta_q_joint": max_target_step_joint,
        "jacobian_singular_values": "METRIC_MISSING_UNTIL_RUNTIME_AUDIT",
        "jacobian_condition_number": "METRIC_MISSING_UNTIL_RUNTIME_AUDIT",
        "dls_lambda": 0.01,
        "dls_lambda_source": "s2_03t_chest_prepose_env_cfg.py DifferentialIKControllerCfg ik_params",
        "old_target_can_pass_0p10_margin_gate": "NO",
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value
