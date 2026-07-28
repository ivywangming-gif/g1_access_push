#!/usr/bin/env python3
"""Validate the frozen S2-03T contract and generate small audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from g1_access_push.stage2.s2_03t_contract import (  # noqa: E402
    ACTION_DIM,
    ACTION_JOINT_NAMES,
    JOINT_LIMIT_MARGIN_RAD,
    LOWER_BODY_ACTION_DIM,
    MAXIMUM_RESIDUAL_CHANGE_RAD,
    OBSERVATION_COMPONENTS,
    OBSERVATION_DIM,
    RESIDUAL_SCALE_RAD,
    REWARD_SPECS,
    TERMINATION_FAILURE_CONTRACT,
    TERMINATION_LIMITS,
    TERMINATION_SUCCESS_CONTRACT,
    TRAINING_CONTRACT_SHA256,
    observation_layout_payload,
    reward_contract_records,
)


CONTRACT_PATH = ROOT / "configs/stage2/s2_03t_contact_training_contract.yaml"
ATTACH_PATH = ROOT / "configs/stage2/s2_03_attach_only.yaml"
RECOMMENDATION_PATH = ROOT / "reports/stage2/training-recommendation.json"
FORMAL_RESULT_PATH = ROOT / "reports/stage2/s2_03_formal_result.json"
PPO_REFERENCE = Path(
    "/root/autodl-tmp/robotics/third_party/WBC-AGILE/agile/rl_env/tasks/pick_place/g1/agents/rsl_rl_ppo_cfg.py"
)
PPO_REFERENCE_SHA256 = "4c28e04a2030b62bb0322f708806e0d9d3f42b31d129961b7d6ca66e486e6a05"
WORKFLOW_DIRECTIVE_SHA256 = "d7e8d53d37c22d85fc2a49acfdba6f05732560fa5b7f7393e67a8e840af361e5"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _source(value: Any, file: str, sha256: str, project_new: bool, rationale: str) -> dict[str, Any]:
    return {
        "value": value,
        "source_file": file,
        "source_sha256": sha256,
        "project_new": project_new,
        "rationale": rationale,
    }


def ppo_provenance() -> dict[str, Any]:
    contract_source = "configs/stage2/s2_03t_contact_training_contract.yaml"
    workflow_source = "/root/.codex/attachments/9b0429d4-e60c-43a1-855c-3e981bc09ab7/pasted-text.txt"
    reference = str(PPO_REFERENCE)
    inherited = "Inherited from the closest installed AGILE G1 manipulation PPO configuration."
    return {
        "schema_version": 1,
        "stage": "S2-03T",
        "reference_config": {"path": reference, "sha256": PPO_REFERENCE_SHA256},
        "critic_observation": "same_77d_as_actor",
        "parameters": {
            "seed": _source(42, contract_source, TRAINING_CONTRACT_SHA256, False, "Frozen clean actor seed."),
            "num_steps_per_env": _source(24, reference, PPO_REFERENCE_SHA256, False, inherited),
            "empirical_normalization": _source(True, reference, PPO_REFERENCE_SHA256, False, inherited),
            "init_noise_std": _source(1.0, reference, PPO_REFERENCE_SHA256, False, inherited),
            "actor_hidden_dims": _source([256, 128, 64], reference, PPO_REFERENCE_SHA256, False, inherited),
            "critic_hidden_dims": _source([256, 128, 64], reference, PPO_REFERENCE_SHA256, False, inherited),
            "activation": _source("elu", reference, PPO_REFERENCE_SHA256, False, inherited),
            "value_loss_coef": _source(1.0, reference, PPO_REFERENCE_SHA256, False, inherited),
            "use_clipped_value_loss": _source(True, reference, PPO_REFERENCE_SHA256, False, inherited),
            "clip_param": _source(0.2, reference, PPO_REFERENCE_SHA256, False, inherited),
            "entropy_coef": _source(0.005, reference, PPO_REFERENCE_SHA256, False, inherited),
            "num_learning_epochs": _source(5, reference, PPO_REFERENCE_SHA256, False, inherited),
            "num_mini_batches": _source(4, reference, PPO_REFERENCE_SHA256, False, inherited),
            "learning_rate": _source(0.001, reference, PPO_REFERENCE_SHA256, False, inherited),
            "schedule": _source("adaptive", reference, PPO_REFERENCE_SHA256, False, inherited),
            "gamma": _source(0.99, reference, PPO_REFERENCE_SHA256, False, inherited),
            "lam": _source(0.95, reference, PPO_REFERENCE_SHA256, False, inherited),
            "desired_kl": _source(0.01, reference, PPO_REFERENCE_SHA256, False, inherited),
            "max_grad_norm": _source(1.0, reference, PPO_REFERENCE_SHA256, False, inherited),
            "pilot_num_envs": _source(64, workflow_source, WORKFLOW_DIRECTIVE_SHA256, True, "Frozen pilot workflow."),
            "pilot_total_iteration_targets": _source(
                [25, 50, 75, 100], workflow_source, WORKFLOW_DIRECTIVE_SHA256, True,
                "Each target is a separate clean run from iteration zero; no resume or checkpoint load.",
            ),
            "formal_num_env_candidates": _source(
                [256, 128, 64], workflow_source, WORKFLOW_DIRECTIVE_SHA256, True,
                "One pre-training capacity smoke may downshift only on OOM.",
            ),
            "formal_max_iterations": _source(
                1000, workflow_source, WORKFLOW_DIRECTIVE_SHA256, True, "Frozen formal execution budget."
            ),
            "formal_save_interval": _source(
                100, workflow_source, WORKFLOW_DIRECTIVE_SHA256, True, "Required checkpoint screening cadence."
            ),
            "upper_actor_recurrent": _source(
                False, workflow_source, WORKFLOW_DIRECTIVE_SHA256, True, "First version explicitly excludes recurrence."
            ),
            "privileged_critic": _source(
                False, workflow_source, WORKFLOW_DIRECTIVE_SHA256, True, "Critic receives the actor's same 77-D input."
            ),
        },
        "forbidden": {
            "resume": True,
            "checkpoint_load": True,
            "pilot_checkpoint_for_formal": True,
            "model_" + "1999": True,
            "recurrent_upper_actor": True,
            "privileged_critic": True,
            "vision": True,
            "curriculum": True,
            "domain_randomization": True,
            "multi_gpu": True,
            "distributed_training": True,
        },
    }


def conflict_report(contract: dict[str, Any], attach: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "S2-03T",
        "status": "RESOLVED_WITH_EXPLICIT_PRECEDENCE",
        "conflict_count": 4,
        "conflicts": [
            {
                "id": "TRAINING_AUTHORIZATION_OVERRIDE",
                "frozen_contract_value": {
                    "training_launch_authorized": contract["promotion_boundary"]["training_launch_authorized"],
                    "training_started": contract["promotion_boundary"]["training_started"],
                    "status": contract["status"],
                },
                "execution_directive": "Run the complete autonomous S2-03T training and qualification workflow.",
                "resolution": "The later explicit execution directive overrides only launch/status authorization. The frozen scientific observation, action, reward, termination, and formal gates remain unchanged.",
            },
            {
                "id": "PILOT_CLOSE_PROCESS_VS_NO_RESUME",
                "frozen_contract_value": {"resume_enabled": contract["initialization"]["resume_enabled"]},
                "execution_directive": "Close training after each 25-iteration pilot block, evaluate separately, and never resume.",
                "resolution": "Use four independent clean no-load/no-resume pilot trainings with total targets 25, 50, 75, and 100 iterations. Every process initializes actor, optimizer, and normalizer from iteration zero; earlier pilot checkpoints are evaluation-only and never loaded.",
                "pilot_total_iteration_targets": [25, 50, 75, 100],
            },
            {
                "id": "CONTACT_TIMEOUT_FAILURE_LIST_VS_TRUNCATION",
                "frozen_contract_value": {
                    "bilateral_contact_timeout_steps": contract["termination_contract"]["failure"]["bilateral_contact_timeout"],
                    "maximum_episode_steps": contract["termination_contract"]["maximum_episode_steps"],
                },
                "execution_directive": "Mark environment timeout as truncation and physical/safety failures as termination.",
                "resolution": "At step 1000, set time_out=true and truncated=true. Record bilateral_contact_timeout independently in extras and its -25 reward term; do not classify it as a physical/safety termination.",
            },
            {
                "id": "S2_03_FORMAL_GATE_COVERAGE",
                "frozen_contract_value": {
                    "training_failure_keys": sorted(contract["termination_contract"]["failure"]),
                    "formal_hand_position_gate_m": attach["acceptance"]["maximum_hand_position_error_m"],
                    "formal_hand_orientation_gate_deg": attach["acceptance"]["maximum_hand_orientation_error_deg"],
                },
                "execution_directive": "Keep every formal S2-03 evaluator threshold unchanged.",
                "resolution": "Training terminations implement exactly the frozen S2-03T YAML and do not silently add the two omitted hand-tracking gates. Checkpoint screening and formal qualification reuse the unchanged S2-03 evaluator; its current implementation records hand-tracking errors but does not promote the two declared attach-config values into additional decision gates.",
            },
        ],
    }


def validate_contract(contract: dict[str, Any]) -> dict[str, bool]:
    observation = contract["observation_contract"]
    expected_components = []
    for component in OBSERVATION_COMPONENTS:
        record: dict[str, Any] = {
            "name": component.name,
            "dimension": component.dimension,
            "unit": component.unit,
        }
        if math_is_symmetric(component.clip_min, component.clip_max):
            record["clip_abs"] = component.clip_max
        else:
            record["clip_min"] = component.clip_min
            record["clip_max"] = component.clip_max
        expected_components.append(record)
    action = contract["action_contract"]
    reward = contract["reward_contract"]
    failure = contract["termination_contract"]["failure"]
    return {
        "contract_sha256": sha256_file(CONTRACT_PATH) == TRAINING_CONTRACT_SHA256,
        "stage": contract.get("stage") == "S2-03T",
        "observation_dim": observation["policy_observation_dim"] == OBSERVATION_DIM,
        "observation_components_exact": observation["components"] == expected_components,
        "observation_finite": observation["finite_required"] is True,
        "action_dim": action["policy_action_dim"] == action["arm_action_dim"] == ACTION_DIM,
        "lower_body_action_dim": action["lower_body_action_dim"] == LOWER_BODY_ACTION_DIM,
        "joint_order": action["ordered_joint_names"] == list(ACTION_JOINT_NAMES),
        "residual_scale": action["residual_scale_rad"] == RESIDUAL_SCALE_RAD,
        "residual_rate": action["maximum_residual_change_per_control_step_rad"] == MAXIMUM_RESIDUAL_CHANGE_RAD,
        "joint_margin": action["joint_limit_margin_rad"] == JOINT_LIMIT_MARGIN_RAD,
        "reward_records_exact": reward["terms"] == reward_contract_records(),
        "no_box_progress_reward": reward["box_progress_reward_enabled"] is False,
        "no_base_progress_reward": reward["base_progress_reward_enabled"] is False,
        "episode_steps": contract["termination_contract"]["maximum_episode_steps"]
        == TERMINATION_LIMITS["maximum_episode_steps"],
        "termination_success_exact": contract["termination_contract"]["success"] == TERMINATION_SUCCESS_CONTRACT,
        "termination_failure_exact": failure == TERMINATION_FAILURE_CONTRACT,
        "termination_limits": all(
            failure[key] == TERMINATION_LIMITS[key]
            for key in (
                "contact_loss_grace_steps",
                "single_hand_maximum_steps",
                "per_palm_force_peak_threshold_n",
                "per_palm_impulse_threshold_ns",
                "excessive_combined_impulse_threshold_ns",
                "contact_force_rate_threshold_nps",
                "maximum_box_linear_speed_mps",
                "maximum_box_angular_speed_radps",
                "maximum_box_translation_m",
                "maximum_box_yaw_change_rad",
                "maximum_base_xy_excursion_m",
                "minimum_root_height_m",
                "maximum_root_height_m",
                "maximum_root_tilt_deg",
                "minimum_arm_joint_limit_margin_rad",
                "maximum_arm_torque_ratio",
            )
        ),
    }


def math_is_symmetric(lower: float, upper: float) -> bool:
    return abs(lower + upper) <= 1.0e-15


def build_artifacts() -> dict[str, dict[str, Any]]:
    if not PPO_REFERENCE.is_file() or sha256_file(PPO_REFERENCE) != PPO_REFERENCE_SHA256:
        raise RuntimeError("installed AGILE G1 pick-place PPO reference is missing or has changed SHA")
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    attach = yaml.safe_load(ATTACH_PATH.read_text(encoding="utf-8"))
    checks = validate_contract(contract)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"frozen S2-03T contract validation failed: {failed}")
    layout = observation_layout_payload()
    layout["training_contract_sha256"] = TRAINING_CONTRACT_SHA256
    ppo = ppo_provenance()
    conflicts = conflict_report(contract, attach)
    resolved = {
        "schema_version": 1,
        "stage": "S2-03T",
        "status": "STATIC_CONTRACT_RESOLVED",
        "scientific_contract_valid": True,
        "validation_checks": checks,
        "source_sha256": {
            "configs/stage2/s2_03t_contact_training_contract.yaml": TRAINING_CONTRACT_SHA256,
            "configs/stage2/s2_03_attach_only.yaml": sha256_file(ATTACH_PATH),
            "reports/stage2/training-recommendation.json": sha256_file(RECOMMENDATION_PATH),
            "reports/stage2/s2_03_formal_result.json": sha256_file(FORMAL_RESULT_PATH),
        },
        "observation": layout,
        "action": {
            "policy_action_dim": ACTION_DIM,
            "arm_action_dim": ACTION_DIM,
            "lower_body_action_dim": LOWER_BODY_ACTION_DIM,
            "ordered_joint_names": list(ACTION_JOINT_NAMES),
            "normalized_range": [-1.0, 1.0],
            "residual_scale_rad": RESIDUAL_SCALE_RAD,
            "maximum_residual_change_per_control_step_rad": MAXIMUM_RESIDUAL_CHANGE_RAD,
            "joint_limit_margin_rad": JOINT_LIMIT_MARGIN_RAD,
            "lower_body_command": contract["initialization"]["lower_body_command"],
            "lower_body_trainable": False,
        },
        "reward_terms": reward_contract_records(),
        "termination": {
            "maximum_episode_steps": TERMINATION_LIMITS["maximum_episode_steps"],
            "timeout_classification": "TRUNCATION",
            "physical_safety_classification": "TERMINATION",
            "success_classification": "INDEPENDENT_SUCCESS_TERM",
            "limits": TERMINATION_LIMITS,
            "success_contract": TERMINATION_SUCCESS_CONTRACT,
            "failure_contract": TERMINATION_FAILURE_CONTRACT,
        },
        "initialization": {
            "seed": 42,
            "clean_actor": True,
            "resume": False,
            "load_checkpoint": False,
            "pilot_checkpoint_used_for_formal": False,
            "model_" + "1999_allowed": False,
        },
        "pilot": {
            "num_envs": 64,
            "baseline_evaluation_episodes": 4,
            "total_iteration_targets": [25, 50, 75, 100],
            "process_semantics": "INDEPENDENT_CLEAN_FROM_ITERATION_ZERO",
            "resume": False,
            "checkpoint_load": False,
            "evaluation_episodes_per_target_maximum": 4,
            "minimum_mean_gap_improvement_m": 0.005,
        },
        "formal": {
            "num_env_candidates_oom_only": [256, 128, 64],
            "maximum_iterations": 1000,
            "save_interval": 100,
            "clean_actor": True,
            "resume": False,
            "pilot_checkpoint_used": False,
        },
        "ppo_parameter_provenance_path": "reports/stage2/s2_03t_ppo_parameter_provenance.json",
        "contract_conflicts_path": "reports/stage2/s2_03t_contract_conflicts.json",
    }
    return {
        "reports/stage2/s2_03t_observation_layout.json": layout,
        "reports/stage2/s2_03t_ppo_parameter_provenance.json": ppo,
        "reports/stage2/s2_03t_contract_conflicts.json": conflicts,
        "reports/stage2/s2_03t_resolved_config.json": resolved,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Compare generated content with committed artifacts.")
    args = parser.parse_args()
    artifacts = build_artifacts()
    if args.check:
        mismatches = []
        for relative, payload in artifacts.items():
            path = ROOT / relative
            expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                mismatches.append(relative)
        if mismatches:
            print(f"S2_03T_RESOLVED_CONFIG=STALE files={mismatches}")
            return 1
        print("S2_03T_RESOLVED_CONFIG=PASS")
        return 0
    for relative, payload in artifacts.items():
        write_json(ROOT / relative, payload)
    print("S2_03T_RESOLVED_CONFIG=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
