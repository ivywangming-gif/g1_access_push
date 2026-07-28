"""Pure static checks for the S2-03T training interface."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from g1_access_push.stage2.s2_03t_contract import (  # noqa: E402
    ACTION_DIM,
    ACTION_JOINT_NAMES,
    AttachEpisodeTracker,
    ContractError,
    OBSERVATION_COMPONENTS,
    OBSERVATION_DIM,
    PHYSICAL_TERMINATION_REASONS,
    REWARD_SPECS,
    TERMINATION_FAILURE_CONTRACT,
    TERMINATION_SUCCESS_CONTRACT,
    TRAINING_CONTRACT_SHA256,
    assemble_observation,
    compute_reward_terms,
    map_arm_residual_action,
    observation_slices,
    validate_result_schema,
)


CONTRACT_PATH = ROOT / "configs/stage2/s2_03t_contact_training_contract.yaml"


def contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def zero_components(batch: int = 1) -> dict[str, list[list[float]]]:
    return {component.name: [[0.0] * component.dimension for _ in range(batch)] for component in OBSERVATION_COMPONENTS}


def safe_signals(**updates: object) -> dict[str, object]:
    signals: dict[str, object] = {
        "left_force_n": 0.0,
        "right_force_n": 0.0,
        "finite": True,
        "forbidden_non_palm_box_collision": False,
        "all_frozen_safety_gates": True,
        "box_linear_speed_mps": 0.0,
        "box_angular_speed_radps": 0.0,
        "box_translation_m": 0.0,
        "box_yaw_change_rad": 0.0,
        "base_xy_excursion_m": 0.0,
        "root_height_m": 0.75,
        "root_tilt_deg": 0.0,
        "minimum_arm_joint_limit_margin_rad": 0.5,
        "arm_torque_ratio_max": 0.5,
    }
    signals.update(updates)
    return signals


def reward_metrics(**updates: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "left_contact": False,
        "right_contact": False,
        "all_frozen_gates": True,
        "frozen_attach_success": False,
        "surface_gap_m": [0.02, 0.03],
        "palm_position_error_m": [0.01, 0.02],
        "palm_orientation_error_rad": [0.1, 0.2],
        "palm_force_norm_n": [1.0, 2.0],
        "box_translation_m": 0.001,
        "box_yaw_change_rad": -0.002,
        "normalized_peak_plus_impulse_plus_rate": 0.25,
        "root_tilt_deg": 1.0,
        "minimum_arm_joint_limit_margin_rad": 0.08,
        "arm_torque_ratio_max": 1.1,
        "forbidden_non_palm_box_collision": False,
        "bilateral_contact_timeout": False,
    }
    metrics.update(updates)
    return metrics


def test_observation_layout_is_exactly_17_components_and_77_dimensions() -> None:
    slices = observation_slices()
    assert len(OBSERVATION_COMPONENTS) == len(slices) == 17
    assert sum(component.dimension for component in OBSERVATION_COMPONENTS) == OBSERVATION_DIM == 77
    cursor = 0
    for component in OBSERVATION_COMPONENTS:
        assert slices[component.name] == slice(cursor, cursor + component.dimension)
        cursor += component.dimension


def test_observation_layout_matches_frozen_yaml_order_units_and_clips() -> None:
    yaml_components = contract()["observation_contract"]["components"]
    assert [item["name"] for item in yaml_components] == [item.name for item in OBSERVATION_COMPONENTS]
    for frozen, implemented in zip(yaml_components, OBSERVATION_COMPONENTS, strict=True):
        assert frozen["dimension"] == implemented.dimension
        assert frozen["unit"] == implemented.unit
        if "clip_abs" in frozen:
            assert implemented.clip_min == -frozen["clip_abs"]
            assert implemented.clip_max == frozen["clip_abs"]
        else:
            assert implemented.clip_min == frozen["clip_min"]
            assert implemented.clip_max == frozen["clip_max"]


def test_observation_is_batch_first_clipped_and_finite() -> None:
    components = zero_components(batch=2)
    components["base_ang_vel_body"][0] = [-9.0, 0.5, 9.0]
    components["palm_actual_surface_gap"][1] = [-1.0, 1.0]
    observation = assemble_observation(components)
    assert len(observation) == 2
    assert all(len(row) == 77 for row in observation)
    assert observation[0][0:3] == [-5.0, 0.5, 5.0]
    gap = observation_slices()["palm_actual_surface_gap"]
    assert observation[1][gap] == [-0.02, 0.10]
    components["previous_arm_action"][0][0] = math.inf
    with pytest.raises(ContractError, match="finite"):
        assemble_observation(components)


def test_action_joint_order_and_dimensions_match_yaml() -> None:
    action = contract()["action_contract"]
    assert ACTION_DIM == action["policy_action_dim"] == action["arm_action_dim"] == 14
    assert action["lower_body_action_dim"] == 0
    assert list(ACTION_JOINT_NAMES) == action["ordered_joint_names"]
    assert all(ACTION_JOINT_NAMES[index].startswith("left_") for index in range(0, 14, 2))
    assert all(ACTION_JOINT_NAMES[index].startswith("right_") for index in range(1, 14, 2))


def test_zero_action_at_reset_is_exact_frozen_reference() -> None:
    reference = [0.1 * (-1 if index % 2 else 1) for index in range(14)]
    mapped = map_arm_residual_action([0.0] * 14, [0.0] * 14, reference, [-2.0] * 14, [2.0] * 14)
    assert mapped.joint_target_rad == tuple(reference)
    assert mapped.applied_residual_rad == (0.0,) * 14


def test_action_clipping_residual_scale_and_rate_limit() -> None:
    first = map_arm_residual_action([2.0, -2.0] * 7, [0.0] * 14, [0.0] * 14, [-1.0] * 14, [1.0] * 14)
    assert first.clipped_action == pytest.approx([1.0, -1.0] * 7)
    assert first.requested_residual_rad == pytest.approx([0.05, -0.05] * 7)
    assert first.applied_residual_rad == pytest.approx([0.005, -0.005] * 7)
    second = map_arm_residual_action(
        [1.0, -1.0] * 7, first.applied_residual_rad, [0.0] * 14, [-1.0] * 14, [1.0] * 14
    )
    assert second.applied_residual_rad == pytest.approx([0.01, -0.01] * 7)


def test_action_target_clamps_to_hard_limit_margin() -> None:
    reference = [-0.895] + [0.0] * 13
    previous = [-0.045] + [0.0] * 13
    mapped = map_arm_residual_action([-1.0] + [0.0] * 13, previous, reference, [-1.0] * 14, [1.0] * 14)
    assert mapped.joint_target_rad[0] == pytest.approx(-0.9)
    assert mapped.minimum_joint_limit_margin_rad == pytest.approx(0.1)
    assert mapped.joint_limit_clamped[0] is True
    with pytest.raises(ContractError, match="frozen IK reference violates"):
        map_arm_residual_action([0.0] * 14, [0.0] * 14, [-0.95] + [0.0] * 13, [-1.0] * 14, [1.0] * 14)


def test_reward_specs_match_yaml_names_weights_and_formulas_exactly() -> None:
    frozen = contract()["reward_contract"]
    implemented = [{"name": item.name, "weight": item.weight, "formula": item.formula} for item in REWARD_SPECS]
    assert implemented == frozen["terms"]
    assert frozen["box_progress_reward_enabled"] is False
    assert frozen["base_progress_reward_enabled"] is False
    assert not any("progress" in item.name for item in REWARD_SPECS)


def test_reward_formulas_and_weighting_are_numeric_and_exact() -> None:
    result = compute_reward_terms(
        reward_metrics(left_contact=True, right_contact=True), [0.5] * 14, [0.25] * 14
    )
    raw = result["raw"]
    weighted = result["weighted"]
    assert raw["bilateral_contact_verify_step"] == 1.0
    assert raw["attached_hold_step"] == 1.0
    assert raw["symmetric_gap_closure"] == pytest.approx(math.exp(-2.5))
    assert raw["hand_position_tracking"] == pytest.approx(0.03)
    assert raw["hand_orientation_tracking"] == pytest.approx(0.3)
    assert raw["force_imbalance"] == pytest.approx(1.0)
    assert raw["box_yaw_change"] == pytest.approx(0.002)
    assert raw["joint_limit_margin"] == pytest.approx(0.02)
    assert raw["torque_ratio"] == pytest.approx(0.1)
    assert raw["action_l2"] == pytest.approx(3.5)
    assert raw["action_rate"] == pytest.approx(0.875)
    weights = {item.name: item.weight for item in REWARD_SPECS}
    assert weighted == pytest.approx({name: weights[name] * value for name, value in raw.items()})
    assert result["total"] == pytest.approx(sum(weighted.values()))


def test_tracker_keeps_left_and_right_contact_and_impulse_separate() -> None:
    tracker = AttachEpisodeTracker()
    snapshot = tracker.step(safe_signals(left_force_n=1.1, right_force_n=0.2))
    state = tracker.state()
    assert snapshot.extras["left_contact"] is True
    assert snapshot.extras["right_contact"] is False
    assert state["left_impulse_ns"] == pytest.approx(1.1 * 0.02)
    assert state["right_impulse_ns"] == pytest.approx(0.2 * 0.02)
    assert state["single_hand_steps"] == 1


def test_termination_success_and_failure_specs_match_yaml_exactly() -> None:
    frozen = contract()["termination_contract"]
    assert frozen["success"] == TERMINATION_SUCCESS_CONTRACT
    assert frozen["failure"] == TERMINATION_FAILURE_CONTRACT
    assert frozen["maximum_episode_steps"] == 1000
    assert frozen["auto_reset_within_evaluation_episode"] is False


def test_tracker_success_counters_require_verify_then_hold() -> None:
    tracker = AttachEpisodeTracker()
    for _ in range(20):
        snapshot = tracker.step(safe_signals(left_force_n=1.1, right_force_n=1.1))
    assert snapshot.success is False
    assert tracker.state()["bilateral_verified"] is True
    assert tracker.state()["attached_hold_steps"] == 0
    for _ in range(100):
        snapshot = tracker.step(safe_signals(left_force_n=1.1, right_force_n=1.1))
    assert snapshot.success is True
    assert snapshot.terminated is False
    assert snapshot.truncated is False


def test_tracker_timeout_is_truncation_not_physical_termination() -> None:
    tracker = AttachEpisodeTracker()
    snapshot = None
    for _ in range(1000):
        snapshot = tracker.step(safe_signals())
    assert snapshot is not None
    assert snapshot.time_out is True
    assert snapshot.truncated is True
    assert snapshot.bilateral_contact_timeout is True
    assert snapshot.terminated is False
    assert snapshot.reasons == ()


def test_tracker_physical_failures_are_individual_termination_fields() -> None:
    tracker = AttachEpisodeTracker()
    snapshot = tracker.step(
        safe_signals(forbidden_non_palm_box_collision=True, box_translation_m=0.006, root_tilt_deg=7.0)
    )
    assert snapshot.terminated is True
    assert snapshot.truncated is False
    assert set(snapshot.reasons) == {
        "FORBIDDEN_NON_PALM_BOX_COLLISION",
        "BOX_TRANSLATION",
        "ROOT_TILT",
    }
    assert set(PHYSICAL_TERMINATION_REASONS).issubset(snapshot.extras)


def test_tracker_reset_clears_contact_impulse_and_counters() -> None:
    tracker = AttachEpisodeTracker()
    tracker.step(safe_signals(left_force_n=1.1, right_force_n=0.0))
    tracker.reset()
    state = tracker.state()
    assert state["step"] == 0
    assert state["left_contact"] is False and state["right_contact"] is False
    assert state["left_impulse_ns"] == state["right_impulse_ns"] == 0.0
    assert state["bilateral_verify_steps"] == state["attached_hold_steps"] == 0
    assert state["single_hand_steps"] == state["contact_loss_steps"] == 0
    assert state["post_initial_reset_count"] == 1


def test_result_schema_rejects_reset_accumulation_and_checkpoint_loading() -> None:
    base = {
        "schema_version": 1,
        "stage": "S2-03T",
        "result_type": "contract_smoke",
        "status": "PASS",
        "primary_reason": "CONTRACT_SMOKE_COMPLETE",
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "observation_shape": [1, 77],
        "action_shape": [1, 14],
        "actor_checkpoint_loaded": False,
        "steps": 3,
    }
    assert validate_result_schema(base) == []
    assert "actor_checkpoint_loaded" in validate_result_schema({**base, "actor_checkpoint_loaded": True})
    formal = {
        **base,
        "result_type": "formal_qualification",
        "episode_count": 1,
        "auto_reset": False,
        "post_initial_reset_count": 0,
    }
    assert validate_result_schema(formal) == []
    assert "post_initial_reset_count" in validate_result_schema({**formal, "post_initial_reset_count": 1})


def test_generated_artifacts_are_current_and_encode_four_conflicts() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/stage2/generate_s2_03t_resolved_config.py"), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    conflicts = json.loads((ROOT / "reports/stage2/s2_03t_contract_conflicts.json").read_text(encoding="utf-8"))
    assert conflicts["conflict_count"] == len(conflicts["conflicts"]) == 4
    ids = {item["id"] for item in conflicts["conflicts"]}
    assert ids == {
        "TRAINING_AUTHORIZATION_OVERRIDE",
        "PILOT_CLOSE_PROCESS_VS_NO_RESUME",
        "CONTACT_TIMEOUT_FAILURE_LIST_VS_TRUNCATION",
        "S2_03_FORMAL_GATE_COVERAGE",
    }
    pilot = next(item for item in conflicts["conflicts"] if item["id"] == "PILOT_CLOSE_PROCESS_VS_NO_RESUME")
    assert pilot["pilot_total_iteration_targets"] == [25, 50, 75, 100]
    assert "iteration zero" in pilot["resolution"]
    resolved = json.loads((ROOT / "reports/stage2/s2_03t_resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["pilot"]["development_evaluation_seeds"] == [42, 43, 44, 45]
    assert resolved["formal"]["screening_development_seeds"] == [42, 43, 44]
    assert resolved["formal"]["qualification_seed"] == 42
