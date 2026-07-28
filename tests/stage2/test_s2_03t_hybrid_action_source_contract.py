"""Dependency-free source contract for the S2-03T hybrid action."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "src/g1_access_push/sim/stage2/s2_03t_actions.py"
CONTRACT_PATH = ROOT / "configs/stage2/s2_03t_contact_action_contract.yaml"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
CONTRACT = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def class_node(name: str) -> ast.ClassDef:
    return next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == name)


def method_source(class_name: str, method_name: str) -> str:
    node = next(
        item
        for item in class_node(class_name).body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name
    )
    value = ast.get_source_segment(SOURCE, node)
    assert value is not None
    return value


def class_defaults(class_name: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in class_node(class_name).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return values


def test_public_action_is_exactly_two_ordered_normal_corrections() -> None:
    action = CONTRACT["actor_action"]
    assert action["dimension"] == 2
    assert action["order"] == [
        "left_palm_normal_correction",
        "right_palm_normal_correction",
    ]
    assert "HYBRID_ACTION_DIM = 2" in SOURCE
    assert "return HYBRID_ACTION_DIM" in method_source("HybridNormalApproachAction", "action_dim")
    assert "ArmResidualAction = HybridNormalApproachAction" in SOURCE
    assert "ArmResidualActionCfg = HybridNormalApproachActionCfg" in SOURCE


def test_reference_stays_14d_while_raw_policy_action_is_2d() -> None:
    initialize = method_source("HybridNormalApproachAction", "__init__")
    set_reference = method_source("HybridNormalApproachAction", "set_reference")
    restore_reference = method_source("HybridNormalApproachAction", "restore_reference")
    assert "action_shape = (self.num_envs, HYBRID_ACTION_DIM)" in initialize
    assert "self._raw_actions = torch.zeros(action_shape" in initialize
    assert "(1, len(ARM_JOINT_NAMES))" in set_reference
    assert "(len(env_ids), len(ARM_JOINT_NAMES))" in restore_reference
    resolve = method_source("HybridNormalApproachAction", "_resolve_env_ids")
    assert "isinstance(env_ids, slice)" in resolve


def test_joint_rate_envelope_is_fixed_at_process_actions_boundary() -> None:
    process = method_source("HybridNormalApproachAction", "process_actions")
    joint = method_source("HybridNormalApproachAction", "_apply_hybrid_joint_target")
    assert "_control_step_joint_origin.copy_(self._processed_actions)" in process
    assert "_control_step_joint_origin" in joint
    assert "_control_step_joint_origin.copy_" not in joint


def test_runtime_defaults_match_frozen_action_contract() -> None:
    cfg = class_defaults("HybridNormalApproachActionCfg")
    action = CONTRACT["actor_action"]
    nominal = CONTRACT["nominal_approach"]
    inverse = CONTRACT["inverse_kinematics"]
    expected = {
        "control_dt_s": CONTRACT["source_facts"]["control_dt_s"],
        "correction_scale_m": action["correction_scale_m"],
        "correction_rate_limit_m_per_control_step": action[
            "correction_rate_limit_m_per_control_step"
        ],
        "contacted_hand_additional_inward_cap_m": action["contacted_hand_additional_inward_cap_m"],
        "nominal_speed_limit_mps": nominal["speed_limit_mps"],
        "nominal_acceleration_limit_mps2": nominal["acceleration_limit_mps2"],
        "nominal_jerk_limit_mps3": nominal["jerk_limit_mps3"],
        "nominal_maximum_displacement_m": nominal["maximum_displacement_m"],
        "dls_damping_lambda": inverse["damping_lambda"],
        "joint_limit_margin_rad": inverse["joint_limit_margin_rad"],
        "joint_target_rate_limit_rad_per_control_step": inverse[
            "joint_target_rate_limit_rad_per_control_step"
        ],
    }
    assert all(cfg[name] == value for name, value in expected.items())
    assert tuple(cfg["palm_orientation_object_wxyz"]) == tuple(
        CONTRACT["source_facts"]["frozen_s2_02_orientation_wxyz"]
    )


def test_nominal_integrates_once_and_apply_only_solves_limited_ik() -> None:
    process = method_source("HybridNormalApproachAction", "process_actions")
    apply = method_source("HybridNormalApproachAction", "apply_actions")
    assert "nominal_jerk_limit_mps3" in process
    assert "nominal_acceleration_limit_mps2" in process
    assert "nominal_speed_limit_mps" in process
    assert "nominal_maximum_displacement_m" in process
    assert "_nominal_displacement_m.copy_" in process
    assert "_nominal_displacement_m" not in apply
    assert "_apply_hybrid_joint_target(0, left_target)" in apply
    assert "_apply_hybrid_joint_target(1, right_target)" in apply
    assert "self._safety_latched[:, None]" in apply
    assert "self._control_step_joint_origin" in apply
    hybrid_branch = apply.index("if not self._bootstrap_mode:")
    unconstrained_left_write = apply.index(
        "set_joint_position_target(left_target, joint_ids=self._left_joint_ids)"
    )
    assert hybrid_branch < unconstrained_left_write


def test_contact_freeze_caps_inward_correction_and_safety_latches() -> None:
    process = method_source("HybridNormalApproachAction", "process_actions")
    safety = method_source("HybridNormalApproachAction", "_update_safety_latch")
    forbidden = method_source("HybridNormalApproachAction", "_filtered_contact_active")
    assert "self._contact_latched |= contacts" in process
    assert "nominal_active = active & ~self._contact_latched" in process
    assert "contacted_hand_additional_inward_cap_m" in process
    assert "_contact_correction_at_latch[newly_contacted]" in process
    assert "torch.minimum(desired_correction, contacted_inward_cap)" in process
    assert "self._safety_latched |= failure" in safety
    for field in (
        "force_peak_threshold_n",
        "palm_impulse_threshold_ns",
        "combined_impulse_threshold_ns",
        "force_rate_threshold_nps",
        "box_translation_threshold_m",
        "box_yaw_threshold_rad",
        "root_tilt_threshold_deg",
        "joint_limit_margin_rad",
        "torque_ratio_threshold",
    ):
        assert field in safety
    assert "_allowed_forbidden_filter_indices" in forbidden


def test_target_uses_frozen_anchor_and_pose_dls_without_mdp_dependency() -> None:
    anchor = method_source("HybridNormalApproachAction", "set_episode_anchor")
    target = method_source("HybridNormalApproachAction", "_set_hybrid_ik_commands")
    joint = method_source("HybridNormalApproachAction", "_apply_hybrid_joint_target")
    assert "_anchor_box_pos_w" in anchor and "_anchor_box_quat_w" in anchor
    assert "_anchor_box_pos_w" in target and "_anchor_box_quat_w" in target
    assert "self._box.data" not in target
    assert 'ik_method="dls"' in SOURCE
    assert 'ik_params={"lambda_val": float(cfg.dls_damping_lambda)}' in SOURCE
    assert "joint_target_rate_limit_rad_per_control_step" in joint
    assert "_target_lower" in joint and "_target_upper" in joint
    assert "s2_03t_mdp" not in SOURCE
    assert "runtime_state" not in SOURCE


def test_frozen_lower_body_action_remains_internal_and_zero_dimensional() -> None:
    lower = ast.get_source_segment(SOURCE, class_node("FrozenRecurrentLowerBodyAction"))
    assert lower is not None
    assert "return 0" in lower
    assert "batched_student_forward" in lower
    assert "output * self._policy_output_scale + self._policy_output_offset" in lower
    assert "BASE_COMMAND = (0.0, 0.0, 0.0, 0.7)" in SOURCE
