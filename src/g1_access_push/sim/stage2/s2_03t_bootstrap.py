"""Deterministically replay the passed S2-03 precontact initialization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

import isaaclab.utils.math as math_utils

from g1_access_push.sim.stage2.s2_03t_actions import ARM_JOINT_NAMES, ArmResidualAction
from g1_access_push.sim.stage2.s2_03t_env import REFERENCE_SCHEMA_VERSION, S203TContactEnv
from g1_access_push.sim.stage2.s2_03t_mdp import (
    DESIRED_PALM_QUATERNION_OBJECT_WXYZ,
    PALM_SUPPORT_OFFSET_M,
    PRECONTACT_GAP_M,
    runtime_state,
)
from g1_access_push.stage2.s2_02_contract import object_local_targets


WARMUP_STEPS = 100
STAND_SETTLE_STEPS = 100
MOVE_TO_PRECONTACT_STEPS = 150
PRECONTACT_HOLD_STEPS = 50
MAXIMUM_POSITION_CORRECTION_M = 0.005
MAXIMUM_ORIENTATION_CORRECTION_RAD = 0.03


def _clamp_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value * torch.clamp(limit / norm.clamp_min(1.0e-9), max=1.0)


def _target_pose_in_pelvis(
    env: S203TContactEnv,
    box_pos: torch.Tensor,
    box_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate = {
        "contact_height_m": 0.62,
        "tangential_separation_m": 0.30,
        "precontact_gap_m": PRECONTACT_GAP_M,
        "palm_collision_support_offset_m": PALM_SUPPORT_OFFSET_M,
    }
    local_positions = torch.tensor(
        object_local_targets(candidate), device=env.device, dtype=torch.float32
    ).expand(env.num_envs, -1, -1)
    desired_q = torch.tensor(
        DESIRED_PALM_QUATERNION_OBJECT_WXYZ, device=env.device, dtype=torch.float32
    ).expand(env.num_envs, 2, -1)
    target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
        box_pos[:, None, :].expand(-1, 2, -1).reshape(-1, 3),
        box_quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
        local_positions.reshape(-1, 3),
        desired_q.reshape(-1, 4),
    )
    robot = env.scene["robot"]
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w[:, None, :].expand(-1, 2, -1).reshape(-1, 3),
        robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
        target_pos_w,
        target_quat_w,
    )
    return (
        target_pos_b.reshape(env.num_envs, 2, 3),
        target_quat_b.reshape(env.num_envs, 2, 4),
    )


def derive_precontact_reference(
    env: S203TContactEnv,
    record_callback: Callable[[str, int, dict[str, torch.Tensor]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay 100+100+150+50 exact S2-03 control steps and capture state."""

    if env.num_envs != 1:
        raise ValueError("authoritative precontact extraction requires exactly one environment")
    env._s2_03t_bootstrap_mode = True
    env.reset(seed=42)
    robot = env.scene["robot"]
    box = env.scene["box"]
    palms = env.scene["hand_frames"]
    arm: ArmResidualAction = env.action_manager.get_term("arm_residual")
    lower = env.action_manager.get_term("frozen_lower_body")
    arm.set_bootstrap_mode(True)
    actions = torch.zeros((1, 14), device=env.device)

    box_state = box.data.default_root_state.clone()
    box_state[:, :3] = torch.stack(
        (
            robot.data.root_link_pos_w[:, 0] + 1.06,
            robot.data.root_link_pos_w[:, 1],
            torch.full((1,), 0.602, device=env.device),
        ),
        dim=-1,
    )
    box_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
    box_state[:, 7:] = 0.0
    box.write_root_state_to_sim(box_state)

    def step_with_desired(desired_pos: torch.Tensor, desired_quat: torch.Tensor, phase: str, step: int) -> None:
        current_pos = palms.data.target_pos_source
        current_quat = palms.data.target_quat_source
        pos_error, orientation_error = math_utils.compute_pose_error(
            current_pos.reshape(-1, 3),
            current_quat.reshape(-1, 4),
            desired_pos.reshape(-1, 3),
            desired_quat.reshape(-1, 4),
            rot_error_type="axis_angle",
        )
        commands = torch.cat(
            (
                _clamp_norm(pos_error.reshape(1, 2, 3), MAXIMUM_POSITION_CORRECTION_M),
                _clamp_norm(
                    orientation_error.reshape(1, 2, 3), MAXIMUM_ORIENTATION_CORRECTION_RAD
                ),
            ),
            dim=-1,
        )
        arm.set_bootstrap_commands(commands)
        _, reward, terminated, truncated, _ = env.step(actions)
        if bool(terminated.any() or truncated.any()):
            raise RuntimeError(f"PRECONTACT_BOOTSTRAP_UNEXPECTED_DONE:{phase}:{step}")
        if not bool(torch.isfinite(reward).all()):
            raise RuntimeError(f"PRECONTACT_BOOTSTRAP_NONFINITE_REWARD:{phase}:{step}")
        if record_callback is not None:
            record_callback(phase, step, runtime_state(env).ensure())
        if (step + 1) % 25 == 0:
            print(f"PHASE=REFERENCE_{phase} step={step + 1}", flush=True)

    zero_commands = torch.zeros((1, 2, 6), device=env.device)
    for step in range(WARMUP_STEPS):
        arm.set_bootstrap_commands(zero_commands)
        _, reward, terminated, truncated, _ = env.step(actions)
        if bool(terminated.any() or truncated.any()) or not bool(torch.isfinite(reward).all()):
            raise RuntimeError(f"PRECONTACT_WARMUP_INVALID:{step}")
        if (step + 1) % 25 == 0:
            print(f"PHASE=REFERENCE_WARMUP step={step + 1}", flush=True)

    baseline_positions = palms.data.target_pos_source.clone()
    baseline_quaternions = palms.data.target_quat_source.clone()
    box_fixed_pos = box.data.root_link_pos_w.clone()
    box_fixed_quat = box.data.root_link_quat_w.clone()
    for step in range(STAND_SETTLE_STEPS):
        step_with_desired(baseline_positions, baseline_quaternions, "STAND_SETTLE", step)
    for step in range(MOVE_TO_PRECONTACT_STEPS):
        target_positions, target_quaternions = _target_pose_in_pelvis(env, box_fixed_pos, box_fixed_quat)
        fraction = float(step + 1) / MOVE_TO_PRECONTACT_STEPS
        desired_positions = baseline_positions + fraction * (target_positions - baseline_positions)
        step_with_desired(desired_positions, target_quaternions, "MOVE", step)
    for step in range(PRECONTACT_HOLD_STEPS):
        target_positions, target_quaternions = _target_pose_in_pelvis(env, box_fixed_pos, box_fixed_quat)
        step_with_desired(target_positions, target_quaternions, "HOLD", step)

    metric = runtime_state(env).ensure()
    if not bool(metric["finite"].all()):
        raise RuntimeError("PRECONTACT_REFERENCE_NONFINITE")
    if bool(metric["contacts"].any() or metric["failure_forbidden_non_palm_box_collision"].any()):
        raise RuntimeError("PRECONTACT_REFERENCE_HAS_COLLISION")
    if bool((metric["arm_joint_margin"] < 0.10).any()):
        raise RuntimeError("PRECONTACT_REFERENCE_JOINT_MARGIN")

    root_state = robot.data.root_state_w[0].detach().cpu().clone()
    root_state[:3] -= env.scene.env_origins[0].detach().cpu()
    box_root_state = box.data.root_state_w[0].detach().cpu().clone()
    box_root_state[:3] -= env.scene.env_origins[0].detach().cpu()
    reference: dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "robot_joint_names": list(robot.joint_names),
        "robot_root_state_relative": root_state,
        "robot_joint_position": robot.data.joint_pos[0].detach().cpu().clone(),
        "robot_joint_velocity": robot.data.joint_vel[0].detach().cpu().clone(),
        "box_root_state_relative": box_root_state,
        "arm_ik_target": arm.bootstrap_targets[0].detach().cpu().clone(),
        **lower.capture_reference(0),
        "source": {
            "stage": "S2-03",
            "source_fail_commit": "e018fe2a3bb284bb51da6a7e245c6772c6e0a87e",
            "source_formal_run": "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/s2_03_formal_20260728_071611",
            "controller_checkpoint_sha256": lower.checkpoint_sha256,
            "steps": {
                "warmup": WARMUP_STEPS,
                "stand_settle": STAND_SETTLE_STEPS,
                "move_to_precontact": MOVE_TO_PRECONTACT_STEPS,
                "precontact_hold": PRECONTACT_HOLD_STEPS,
            },
        },
    }
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "arm_actual_joint_position": robot.data.joint_pos[0, arm.joint_ids].detach().cpu().tolist(),
        "arm_ik_target": reference["arm_ik_target"].tolist(),
        "maximum_actual_to_ik_target_error_rad": float(
            torch.max(
                torch.abs(
                    robot.data.joint_pos[0, arm.joint_ids]
                    - arm.bootstrap_targets[0]
                )
            )
        ),
        "surface_gap_m": metric["gaps"][0].detach().cpu().tolist(),
        "palm_force_n": metric["forces"][0].detach().cpu().tolist(),
        "root_height_m": float(metric["root_height"][0]),
        "root_tilt_deg": float(metric["root_tilt_deg"][0]),
        "minimum_arm_joint_margin_rad": float(metric["arm_joint_margin"][0]),
        "maximum_arm_torque_ratio": float(metric["arm_torque_ratio"][0]),
        "finite": True,
        "contact_free": True,
        "source": reference["source"],
    }
    arm.set_bootstrap_mode(False)
    env._s2_03t_bootstrap_mode = False
    return reference, audit
