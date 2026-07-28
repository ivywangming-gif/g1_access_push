"""Pure contract tests for S2-02 development search and formal evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

from g1_access_push.stage2.s2_02_contract import (
    INVALID_REASONS,
    VALID_FAIL_REASONS,
    candidate_static_checks,
    classify_formal,
    load_config,
    object_local_targets,
    resolved_config,
    score_candidate,
    select_candidate,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stage2/s2_02_precontact_audit.yaml"
SELECTION_REPORT = ROOT / "reports/stage2/s2_02_selected_candidate.json"


def cfg() -> dict:
    return load_config(CONFIG)


def test_scope_and_s2_01_pass_provenance_are_frozen() -> None:
    config = cfg()
    assert config["development_scope"] == "SIMULATION_DEVELOPMENT_ONLY"
    assert config["certification"]["s2_01_pass_commit"] == "e5149a3c3098f8c1a04ef4eafd5a86cd1110456b"
    assert config["object"]["baseline_id"] == "S2_LIGHT_BOX_DEVELOPMENT_BASELINE_V1"


def test_same_s2_01_box_mass_low_com_and_inertia() -> None:
    obj = cfg()["object"]
    assert obj["size_xyz_m"] == [1.2, 0.6, 1.2]
    assert obj["mass_kg"] == 5.0
    assert obj["center_of_mass_local_xyz_m"] == [0.0, 0.0, -0.4]
    assert obj["diagonal_inertia_kg_m2"] == [
        0.12083333333333333, 0.43333333333333335, 0.5208333333333334,
    ]


def test_candidate_count_bound_and_all_static_geometry() -> None:
    config = cfg()
    candidates = config["search"]["candidates"]
    assert len(candidates) == 16 <= config["search"]["maximum_candidates"] == 40
    assert all(all(candidate_static_checks(candidate, config["object"]).values()) for candidate in candidates)


def test_object_targets_are_mirrored_equal_height_and_outside_rear() -> None:
    for candidate in cfg()["search"]["candidates"]:
        left, right = object_local_targets(candidate)
        assert left[0] == right[0] < -0.6
        assert left[1] == -right[1]
        assert left[2] == right[2]


def test_runtime_palm_support_offset_is_applied_before_surface_gap() -> None:
    candidate = {
        "contact_height_m": 0.62,
        "tangential_separation_m": 0.30,
        "precontact_gap_m": 0.06,
        "palm_collision_support_offset_m": 0.32,
    }
    left, right = object_local_targets(candidate)
    assert left[0] == right[0] == -0.98


def test_palm_local_plus_z_quaternion_maps_to_object_plus_x() -> None:
    controller = cfg()["controller"]
    assert controller["palm_local_normal_axis"] == [0.0, 0.0, 1.0]
    w, x, y, z = controller["desired_palm_quaternion_in_object_wxyz"]
    mapped_x = 2.0 * (x * z + w * y)
    mapped_y = 2.0 * (y * z - w * x)
    mapped_z = 1.0 - 2.0 * (x * x + y * y)
    assert [mapped_x, mapped_y, mapped_z] == [1.0000000000000002, 0.0, -2.220446049250313e-16]


def test_development_search_candidate_is_frozen_and_formal_runnable() -> None:
    config = cfg()
    selection = config["selection"]
    assert selection == {
        "status": "FROZEN", "candidate_index": 9,
        "contact_height_m": 0.62, "tangential_separation_m": 0.30,
        "base_to_box_center_distance_m": 1.06, "precontact_gap_m": 0.06,
        "desired_palm_quaternion_in_object_wxyz": [
            0.7071067811865476, 0.0, 0.7071067811865476, 0.0,
        ],
        "palm_collision_support_offset_m": 0.44023889869451527,
    }
    evidence = json.loads(SELECTION_REPORT.read_text(encoding="utf-8"))
    selected = evidence["selected_candidate"]
    assert evidence["status"] == "PASS"
    assert selected["passed"] is True
    assert selected["candidate_index"] == selection["candidate_index"]
    for key in (
        "contact_height_m", "tangential_separation_m",
        "base_to_box_center_distance_m", "precontact_gap_m",
        "palm_collision_support_offset_m",
    ):
        assert selected["candidate"][key] == selection[key]
    assert selected["desired_palm_quaternion_in_object_wxyz"] == selection["desired_palm_quaternion_in_object_wxyz"]
    resolved = resolved_config(CONFIG)
    assert resolved["runnable"] is True
    assert resolved["qualification_state"] == "READY_FOR_PREFLIGHT_AND_FORMAL"


def test_candidate_scoring_rejects_any_hard_gate_and_prefers_margin() -> None:
    def record(index: int, margin: float, passed: bool = True) -> dict:
        checks = {name: passed for name in (
            "finite", "no_contact", "no_overlap", "reachable", "normal_alignment",
            "joint_margin", "symmetric_tracking", "gap_clear", "arm_tracking", "arm_torque",
            "robot_stable", "box_still",
        )}
        return {
            "candidate_index": index,
            "checks": checks,
            "metrics": {
                "minimum_arm_joint_limit_margin_rad": margin,
                "left_right_position_error_asymmetry_m": 0.001,
                "maximum_hand_position_error_m": 0.01,
                "arm_displacement_norm_rad": 0.5,
            },
        }
    low = record(1, 0.2)
    high = record(2, 0.4)
    invalid = record(3, 1.0, False)
    assert score_candidate(invalid)[0] == 0.0
    assert select_candidate([low, invalid, high]) is high


def valid_record() -> dict:
    return {
        "finite": True,
        "robot_box_contact_force_n": 0.0,
        "robot_box_overlap_count": 0,
        "left_position_error_m": 0.005,
        "right_position_error_m": 0.006,
        "left_orientation_error_deg": 1.0,
        "right_orientation_error_deg": 1.2,
        "minimum_arm_joint_limit_margin_rad": 0.3,
        "arm_joint_target_error_max_rad": 0.01,
        "arm_torque_ratio_max": 0.4,
        "base_xy_excursion_m": 0.01,
        "root_height_m": 0.75,
        "root_tilt_deg": 2.0,
        "box_translation_m": 0.001,
        "box_yaw_change_rad": 0.001,
        "box_linear_speed_mps": 0.001,
        "box_angular_speed_radps": 0.001,
        "left_palm_normal_alignment_dot": 0.999,
        "right_palm_normal_alignment_dot": 0.998,
        "left_actual_gap_m": 0.06,
        "right_actual_gap_m": 0.061,
        "commanded_precontact_gap_m": 0.06,
        "post_initial_reset_count": 0,
    }


def test_valid_complete_formal_is_pass() -> None:
    records = [valid_record() for _ in range(cfg()["formal"]["expected_frames"])]
    result = classify_formal(cfg(), records, final_image_present=True)
    assert result == {"status": "PASS", "primary_reason": "ALL_S2_02_GATES_PASSED", "all_reasons": []}


def test_contact_is_valid_physical_fail_not_invalid() -> None:
    records = [valid_record() for _ in range(cfg()["formal"]["expected_frames"])]
    records[-1]["robot_box_contact_force_n"] = 0.1
    result = classify_formal(cfg(), records, final_image_present=True)
    assert result["status"] == "FAIL"
    assert result["primary_reason"] == "UNEXPECTED_ROBOT_BOX_CONTACT"


def test_incomplete_or_nonfinite_evidence_is_invalid() -> None:
    assert classify_formal(cfg(), [], final_image_present=True)["primary_reason"] == "INCOMPLETE_TRACE"
    records = [valid_record() for _ in range(cfg()["formal"]["expected_frames"])]
    records[10]["finite"] = False
    assert classify_formal(cfg(), records, final_image_present=True)["primary_reason"] == "NONFINITE"


def test_fail_and_invalid_reasons_are_disjoint() -> None:
    assert VALID_FAIL_REASONS.isdisjoint(INVALID_REASONS)


def test_frozen_thresholds_are_not_looser_than_stage1_hand_gates() -> None:
    acceptance = cfg()["acceptance"]
    assert acceptance["maximum_hand_position_p95_m"] == 0.03
    assert acceptance["maximum_hand_position_max_m"] == 0.06
    assert acceptance["maximum_hand_orientation_p95_deg"] == 5.0
    assert acceptance["maximum_hand_orientation_max_deg"] == 8.0
    assert math.isclose(acceptance["minimum_palm_normal_alignment_dot"], math.cos(math.radians(10.0)))


def test_prohibitions_are_explicit() -> None:
    assert cfg()["prohibitions"] == {
        "contact": False, "attach": False, "pushing": False, "training": False,
        "planner": False, "base_motion_command": False, "model_1999": False,
    }
