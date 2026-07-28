"""Pure contract tests for S2-03 parameter freezing and classification."""

from __future__ import annotations

import json
from pathlib import Path

from g1_access_push.stage2.s2_03_contract import advance_rate_limited, classify_formal, load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stage2/s2_03_attach_only.yaml"
RESOLVED = ROOT / "reports/stage2/s2_03_resolved_config.json"
PROVENANCE = ROOT / "reports/stage2/s2_03_parameter_provenance.json"


def config(): return load_config(CONFIG)


def test_parameter_provenance_is_pass_and_config_is_runnable() -> None:
    provenance = json.loads(PROVENANCE.read_text())
    resolved = json.loads(RESOLVED.read_text())
    assert provenance["status"] == "PASS" and all(provenance["checks"].values())
    assert resolved["runnable"] is True
    derived = provenance["derived_parameters"]
    assert derived["no_contact_noise_upper_bound_n"] == 0.0
    assert derived["contact_force_threshold_n"] == 1.0
    assert derived["control_frequency_hz"] == 50.0
    assert derived["precontact_gap_m"] == 0.06
    assert derived["contact_impulse_window_steps"] == 5


def test_s2_02_pose_and_no_push_contract_are_frozen() -> None:
    cfg = config()
    assert cfg["certification"]["s2_02_pass_commit"] == "2f44b43d2678c4a882104be97fd6d7dff4bd24b8"
    assert cfg["precontact"]["candidate_index"] == 9
    assert cfg["precontact"]["precontact_gap_m"] == 0.06
    assert cfg["prohibitions"]["pushing"] is False
    assert cfg["prohibitions"]["box_displacement_subgoal"] == [0.0, 0.0, 0.0]
    assert cfg["prohibitions"]["base_motion_command"] == [0.0, 0.0, 0.0]


def test_rate_limiter_respects_speed_acceleration_and_jerk() -> None:
    cfg = config(); m = cfg["motion"]; dt = cfg["controller"]["control_dt_s"]
    displacement = speed = acceleration = 0.0
    previous_acceleration = 0.0
    for _ in range(400):
        displacement, speed, acceleration = advance_rate_limited(
            displacement, speed, acceleration, dt_s=dt,
            speed_limit_mps=m["approach_speed_mps"],
            acceleration_limit_mps2=m["approach_acceleration_limit_mps2"],
            jerk_limit_mps3=m["approach_jerk_limit_mps3"],
        )
        assert speed <= m["approach_speed_mps"] + 1e-12
        assert acceleration <= m["approach_acceleration_limit_mps2"] + 1e-12
        assert abs(acceleration - previous_acceleration) <= m["approach_jerk_limit_mps3"] * dt + 1e-12
        previous_acceleration = acceleration
    assert displacement >= m["maximum_approach_distance_m"]


def valid_record(frame: int, state: str) -> dict:
    return {
        "frame": frame, "fsm_state": state, "finite": True,
        "left_contact": state == "ATTACHED_HOLD", "right_contact": state == "ATTACHED_HOLD",
        "left_force_n": 2.0, "right_force_n": 2.0, "left_impulse_ns": 0.1,
        "right_impulse_ns": 0.1, "combined_impulse_ns": 0.2, "contact_force_peak_n": 2.0,
        "contact_force_rate_nps": 10.0, "forbidden_contact_links": [], "box_translation_m": 0.001,
        "box_yaw_change_rad": 0.001, "box_linear_speed_mps": 0.001, "box_angular_speed_radps": 0.001,
        "root_height_m": 0.73, "root_tilt_deg": 3.0, "minimum_arm_joint_limit_margin_rad": 0.5,
        "arm_torque_ratio_max": 0.5, "post_initial_reset_count": 0,
    }


def test_complete_fsm_hold_is_pass() -> None:
    states = ["STAND_SETTLE", "PRECONTACT", "APPROACH_NORMAL", "BILATERAL_CONTACT_VERIFY"]
    records = [valid_record(i, state) for i, state in enumerate(states)]
    records += [valid_record(len(records) + i, "ATTACHED_HOLD") for i in range(config()["motion"]["attached_hold_steps"])]
    result = classify_formal(config(), records, {"terminal_state": "PASS", "failure_reason": None}, final_image_present=True)
    assert result["status"] == "PASS"


def test_forbidden_contact_is_valid_physical_fail() -> None:
    record = valid_record(0, "APPROACH_NORMAL")
    record["forbidden_contact_links"] = ["/World/envs/env_0/Robot/left_hand/left_hand_index_1_link"]
    result = classify_formal(config(), [record], {"terminal_state": "FAIL", "failure_reason": "FORBIDDEN_BODY_BOX_COLLISION"}, final_image_present=True)
    assert result == {"status": "FAIL", "primary_reason": "FORBIDDEN_BODY_BOX_COLLISION", "all_reasons": ["FORBIDDEN_BODY_BOX_COLLISION"]}


def test_missing_image_or_nonfinite_is_invalid() -> None:
    record = valid_record(0, "APPROACH_NORMAL")
    assert classify_formal(config(), [record], {"terminal_state": "PASS"}, final_image_present=False)["status"] == "INVALID"
    record["finite"] = False
    assert classify_formal(config(), [record], {"terminal_state": "PASS"}, final_image_present=True)["primary_reason"] == "NONFINITE"
