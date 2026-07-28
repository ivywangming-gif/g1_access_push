"""Static contract checks for the unstarted S2-03T training recommendation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RECOMMENDATION = ROOT / "reports/stage2/training-recommendation.json"
CONTRACT = ROOT / "configs/stage2/s2_03t_contact_training_contract.yaml"
ATTACH = ROOT / "configs/stage2/s2_03_attach_only.yaml"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_training_is_justified_but_not_started() -> None:
    recommendation = json.loads(RECOMMENDATION.read_text(encoding="utf-8"))
    assert recommendation["decision"] == "JUSTIFIED_NOT_STARTED"
    assert recommendation["contact_training_justified"] is True
    assert recommendation["training_started"] is False
    evidence = recommendation["evidence"]
    assert evidence["s2_03_status"] == "FAIL"
    assert evidence["primary_reason"] == "BILATERAL_CONTACT_TIMEOUT"
    assert evidence["valid_physical_result"] is True
    assert evidence["precontact_status"] == "PASS"
    assert evidence["robot_stability"] == "PASS"
    assert evidence["implementation_error"] is False


def test_observation_and_action_dimensions_are_exact() -> None:
    contract = load_yaml(CONTRACT)
    observation = contract["observation_contract"]
    assert sum(component["dimension"] for component in observation["components"]) == 77
    assert observation["policy_observation_dim"] == 77
    action = contract["action_contract"]
    assert action["policy_action_dim"] == action["arm_action_dim"] == 14
    assert action["lower_body_action_dim"] == 0
    assert len(action["ordered_joint_names"]) == len(set(action["ordered_joint_names"])) == 14
    assert action["lower_body_command_source"] == "fixed_internal"


def test_clean_initialization_and_scope_boundaries_are_frozen() -> None:
    contract = load_yaml(CONTRACT)
    init = contract["initialization"]
    scope = contract["scope"]
    assert init["clean_actor_initialization"] is True
    assert init["resume_enabled"] is False
    assert init["old_pilot_actor_allowed"] is False
    assert init["model_1999_allowed"] is False
    assert init["lower_body_controller_trainable"] is False
    assert scope["pushing_enabled"] is False
    assert scope["planner_enabled"] is False
    assert scope["whole_task_training_enabled"] is False
    assert contract["promotion_boundary"]["training_launch_authorized"] is False
    assert contract["promotion_boundary"]["training_started"] is False


def test_scientific_termination_thresholds_match_frozen_s2_03() -> None:
    contract = load_yaml(CONTRACT)
    attach = load_yaml(ATTACH)
    success = contract["termination_contract"]["success"]
    failure = contract["termination_contract"]["failure"]
    contact = attach["contact_gate"]
    acceptance = attach["acceptance"]
    assert success["bilateral_contact_threshold_n"] == [
        contact["left_contact_force_threshold_n"], contact["right_contact_force_threshold_n"]
    ]
    assert success["bilateral_verification_steps"] == attach["motion"]["bilateral_verification_steps"]
    assert success["attached_hold_steps"] == attach["motion"]["attached_hold_steps"]
    mapping = {
        "per_palm_force_peak_threshold_n": contact["per_palm_force_peak_threshold_n"],
        "per_palm_impulse_threshold_ns": contact["per_palm_impulse_threshold_ns"],
        "excessive_combined_impulse_threshold_ns": contact["excessive_combined_impulse_threshold_ns"],
        "contact_force_rate_threshold_nps": contact["contact_force_rate_threshold_nps"],
        "maximum_box_translation_m": acceptance["maximum_box_translation_m"],
        "maximum_box_yaw_change_rad": acceptance["maximum_box_yaw_change_rad"],
        "maximum_root_tilt_deg": acceptance["maximum_root_tilt_deg"],
        "minimum_arm_joint_limit_margin_rad": acceptance["minimum_arm_joint_limit_margin_rad"],
        "maximum_arm_torque_ratio": acceptance["maximum_arm_torque_ratio"],
    }
    assert all(failure[key] == value for key, value in mapping.items())


def test_reward_cannot_incentivize_box_or_base_progress() -> None:
    reward = load_yaml(CONTRACT)["reward_contract"]
    assert reward["box_progress_reward_enabled"] is False
    assert reward["base_progress_reward_enabled"] is False
    weights = {term["name"]: term["weight"] for term in reward["terms"]}
    formulas = {term["name"]: term["formula"] for term in reward["terms"]}
    assert weights["box_translation"] < 0.0
    assert weights["box_yaw_change"] < 0.0
    assert formulas["symmetric_gap_closure"] == "exp_minus_50_times_sum_abs_surface_gap"
