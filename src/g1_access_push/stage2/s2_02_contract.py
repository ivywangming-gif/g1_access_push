"""Pure S2-02 configuration, candidate scoring, and evidence classification."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


VALID_FAIL_REASONS = {
    "NO_FEASIBLE_PRECONTACT_GEOMETRY",
    "UNEXPECTED_ROBOT_BOX_CONTACT",
    "FORBIDDEN_ROBOT_BOX_OVERLAP",
    "PRECONTACT_TARGET_TRACKING_FAILED",
    "PALM_NORMAL_ALIGNMENT_FAILED",
    "ARM_JOINT_MARGIN_FAILED",
    "ARM_JOINT_TRACKING_FAILED",
    "ARM_TORQUE_LIMIT_FAILED",
    "ROBOT_STABILITY_FAILED",
    "BOX_STILLNESS_FAILED",
    "AUTO_RESET_DETECTED",
}

INVALID_REASONS = {
    "IMPLEMENTATION_EXCEPTION",
    "MISSING_REQUIRED_FIELD",
    "INCOMPLETE_TRACE",
    "NONFINITE",
    "CONFIG_SHA_MISMATCH",
    "CHECKPOINT_CONTRACT_FAILED",
    "MULTIPLE_ISAAC_PROCESSES",
    "MISSING_FINAL_IMAGE",
}


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("stage") != "S2-02":
        raise ValueError("not an S2-02 config")
    if len(data["search"]["candidates"]) > int(data["search"]["maximum_candidates"]):
        raise ValueError("candidate count exceeds frozen maximum")
    return data


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def object_local_targets(candidate: dict[str, float], rear_face_x: float = -0.6) -> list[list[float]]:
    """Return exact mirrored left/right palm origins in the box frame."""
    separation = float(candidate["tangential_separation_m"])
    height = float(candidate["contact_height_m"])
    gap = float(candidate["precontact_gap_m"])
    support = float(candidate.get("palm_collision_support_offset_m", 0.0))
    x = float(rear_face_x) - gap - support
    z = -0.6 + height
    return [[x, separation / 2.0, z], [x, -separation / 2.0, z]]


def candidate_static_checks(candidate: dict[str, float], object_cfg: dict[str, Any]) -> dict[str, bool]:
    targets = object_local_targets(candidate, float(object_cfg["rear_face_xO_m"]))
    half_y = float(object_cfg["size_xyz_m"][1]) / 2.0
    height = float(candidate["contact_height_m"])
    return {
        "outside_rear_face": all(point[0] < float(object_cfg["rear_face_xO_m"]) for point in targets),
        "mirrored_y": math.isclose(targets[0][1], -targets[1][1], abs_tol=1.0e-12),
        "equal_height": math.isclose(targets[0][2], targets[1][2], abs_tol=1.0e-12),
        "inside_face_tangent_extent": all(abs(point[1]) < half_y for point in targets),
        "inside_face_height_extent": 0.0 < height < float(object_cfg["size_xyz_m"][2]),
        "positive_gap": float(candidate["precontact_gap_m"]) > 0.0,
        "positive_base_distance": float(candidate["base_to_box_center_distance_m"]) > 0.6,
    }


def score_candidate(record: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic priority exactly follows the S2-02 research-lead contract."""
    checks = record["checks"]
    hard_ok = all(checks[name] for name in (
        "finite", "no_contact", "no_overlap", "reachable", "normal_alignment",
        "joint_margin", "symmetric_tracking", "gap_clear", "arm_tracking",
        "arm_torque", "robot_stable", "box_still",
    ))
    metrics = record["metrics"]
    return (
        1.0 if hard_ok else 0.0,
        float(metrics.get("minimum_arm_joint_limit_margin_rad", -math.inf)),
        -float(metrics.get("left_right_position_error_asymmetry_m", math.inf)),
        -float(metrics.get("maximum_hand_position_error_m", math.inf)),
        -float(metrics.get("arm_displacement_norm_rad", math.inf)),
        -float(record["candidate_index"]),
    )


def select_candidate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [record for record in records if score_candidate(record)[0] == 1.0]
    return max(valid, key=score_candidate) if valid else None


def resolved_config(path: Path) -> dict[str, Any]:
    config = load_config(path)
    selection = config["selection"]
    runnable = (
        selection["status"] == "FROZEN"
        and selection["candidate_index"] is not None
        and selection["palm_collision_support_offset_m"] is not None
    )
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "stage": "S2-02",
        "runnable": runnable,
        "qualification_state": "READY_FOR_PREFLIGHT_AND_FORMAL" if runnable else "READY_FOR_DEVELOPMENT_SEARCH",
        "config_sha256": sha256_file(path),
        "contract_digest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "resolved": config,
    }


def classify_formal(config: dict[str, Any], records: list[dict[str, Any]], *, final_image_present: bool) -> dict[str, Any]:
    expected = int(config["formal"]["expected_frames"])
    if len(records) != expected:
        return {"status": "INVALID", "primary_reason": "INCOMPLETE_TRACE", "all_reasons": ["INCOMPLETE_TRACE"]}
    if not final_image_present:
        return {"status": "INVALID", "primary_reason": "MISSING_FINAL_IMAGE", "all_reasons": ["MISSING_FINAL_IMAGE"]}
    required = {
        "finite", "robot_box_contact_force_n", "robot_box_overlap_count", "left_position_error_m",
        "right_position_error_m", "left_orientation_error_deg", "right_orientation_error_deg",
        "minimum_arm_joint_limit_margin_rad", "arm_joint_target_error_max_rad", "arm_torque_ratio_max",
        "base_xy_excursion_m", "root_height_m", "root_tilt_deg", "box_translation_m",
        "box_yaw_change_rad", "box_linear_speed_mps", "box_angular_speed_radps",
        "left_palm_normal_alignment_dot", "right_palm_normal_alignment_dot", "post_initial_reset_count",
        "left_actual_gap_m", "right_actual_gap_m", "commanded_precontact_gap_m",
    }
    if any(not required.issubset(record) for record in records):
        return {"status": "INVALID", "primary_reason": "MISSING_REQUIRED_FIELD", "all_reasons": ["MISSING_REQUIRED_FIELD"]}
    if not all(record["finite"] for record in records):
        return {"status": "INVALID", "primary_reason": "NONFINITE", "all_reasons": ["NONFINITE"]}
    acceptance = config["acceptance"]
    reasons: list[str] = []
    if max(record["robot_box_contact_force_n"] for record in records) > acceptance["maximum_robot_box_contact_force_n"]:
        reasons.append("UNEXPECTED_ROBOT_BOX_CONTACT")
    if max(record["robot_box_overlap_count"] for record in records) > acceptance["maximum_robot_box_overlap_count"]:
        reasons.append("FORBIDDEN_ROBOT_BOX_OVERLAP")
    hold_start = int(config["formal"]["stand_settle_steps"] + config["formal"]["move_to_precontact_steps"])
    hold = records[hold_start:]
    hand_errors = [r[side] for r in hold for side in ("left_position_error_m", "right_position_error_m")]
    sorted_errors = sorted(hand_errors)
    p95 = sorted_errors[min(len(sorted_errors) - 1, math.ceil(0.95 * len(sorted_errors)) - 1)]
    if p95 > acceptance["maximum_hand_position_p95_m"] or max(hand_errors) > acceptance["maximum_hand_position_max_m"]:
        reasons.append("PRECONTACT_TARGET_TRACKING_FAILED")
    if min(min(r["left_palm_normal_alignment_dot"], r["right_palm_normal_alignment_dot"]) for r in hold) < acceptance["minimum_palm_normal_alignment_dot"]:
        reasons.append("PALM_NORMAL_ALIGNMENT_FAILED")
    orientation_errors = [r[side] for r in hold for side in ("left_orientation_error_deg", "right_orientation_error_deg")]
    sorted_orientation = sorted(orientation_errors)
    orientation_p95 = sorted_orientation[min(len(sorted_orientation) - 1, math.ceil(0.95 * len(sorted_orientation)) - 1)]
    if orientation_p95 > acceptance["maximum_hand_orientation_p95_deg"] or max(orientation_errors) > acceptance["maximum_hand_orientation_max_deg"]:
        reasons.append("PRECONTACT_TARGET_TRACKING_FAILED")
    if (min(min(r["left_actual_gap_m"], r["right_actual_gap_m"]) for r in hold) < acceptance["minimum_actual_precontact_gap_m"]
            or max(max(abs(r["left_actual_gap_m"] - r["commanded_precontact_gap_m"]), abs(r["right_actual_gap_m"] - r["commanded_precontact_gap_m"])) for r in hold) > acceptance["maximum_actual_gap_error_m"]):
        reasons.append("PRECONTACT_TARGET_TRACKING_FAILED")
    if min(r["minimum_arm_joint_limit_margin_rad"] for r in records) < acceptance["minimum_arm_joint_limit_margin_rad"]:
        reasons.append("ARM_JOINT_MARGIN_FAILED")
    arm_errors = sorted(r["arm_joint_target_error_max_rad"] for r in records)
    arm_p95 = arm_errors[min(len(arm_errors) - 1, math.ceil(0.95 * len(arm_errors)) - 1)]
    if arm_p95 > acceptance["maximum_arm_joint_target_p95_error_rad"]:
        reasons.append("ARM_JOINT_TRACKING_FAILED")
    if max(r["arm_torque_ratio_max"] for r in records) > acceptance["maximum_arm_torque_ratio"]:
        reasons.append("ARM_TORQUE_LIMIT_FAILED")
    if (max(r["base_xy_excursion_m"] for r in records) > acceptance["maximum_base_xy_excursion_m"]
            or min(r["root_height_m"] for r in records) < acceptance["minimum_root_height_m"]
            or max(r["root_height_m"] for r in records) > acceptance["maximum_root_height_m"]
            or max(r["root_tilt_deg"] for r in records) > acceptance["maximum_root_tilt_deg"]):
        reasons.append("ROBOT_STABILITY_FAILED")
    post_settle = records[int(config["formal"]["stand_settle_steps"]):]
    if (max(r["box_translation_m"] for r in post_settle) > acceptance["maximum_box_translation_m"]
            or max(r["box_yaw_change_rad"] for r in post_settle) > acceptance["maximum_box_yaw_change_rad"]
            or max(r["box_linear_speed_mps"] for r in post_settle) > acceptance["maximum_box_linear_speed_mps"]
            or max(r["box_angular_speed_radps"] for r in post_settle) > acceptance["maximum_box_angular_speed_radps"]):
        reasons.append("BOX_STILLNESS_FAILED")
    if max(r["post_initial_reset_count"] for r in records) != 0:
        reasons.append("AUTO_RESET_DETECTED")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "primary_reason": "ALL_S2_02_GATES_PASSED" if not reasons else reasons[0],
        "all_reasons": reasons,
    }
