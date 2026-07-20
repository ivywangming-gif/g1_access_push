"""Run a deterministic single-arm motion test in the Stage-1 no-box environment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output_json", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402

from g1_access_push.sim.stage1.no_box_env_cfg import (  # noqa: E402
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    G1Stage1NoBoxEnvCfg,
)


def term_slice(names: list[str], dims: list[int], name: str) -> slice:
    index = names.index(name)
    start = sum(dims[:index])
    return slice(start, start + dims[index])


def p95(values: torch.Tensor) -> float:
    return float(torch.quantile(values.flatten(), 0.95).item())


def quaternion_error_deg(
    quaternions: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    dots = torch.sum(quaternions * reference, dim=-1).abs()
    dots = torch.clamp(dots, 0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dots))


def main() -> None:
    config = yaml.safe_load(args.config.read_text())

    active_side = config["active_side"]
    if active_side not in {"left", "right"}:
        raise ValueError("active_side must be 'left' or 'right'")

    inactive_side = "right" if active_side == "left" else "left"

    active_joint_names = LEFT_ARM_JOINT_NAMES if active_side == "left" else RIGHT_ARM_JOINT_NAMES
    inactive_joint_names = RIGHT_ARM_JOINT_NAMES if active_side == "left" else LEFT_ARM_JOINT_NAMES

    cfg = G1Stage1NoBoxEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = int(config["seed"])
    cfg.sim.device = args.device

    env = ManagerBasedEnv(cfg=cfg)
    env.reset(seed=cfg.seed)

    robot = env.scene["robot"]
    hand_frames = env.scene["hand_frames"]

    action_names = env.action_manager.active_terms
    action_dims = env.action_manager.action_term_dim

    upper_slice = term_slice(
        action_names,
        action_dims,
        "upper_body_joint_pos",
    )
    lower_slice = term_slice(
        action_names,
        action_dims,
        "lower_body_joint_pos",
    )

    upper_term = env.action_manager.get_term("upper_body_joint_pos")
    upper_joint_names = list(upper_term.IO_descriptor.joint_names)

    expected_upper_dim = upper_slice.stop - upper_slice.start
    if len(upper_joint_names) != expected_upper_dim:
        raise RuntimeError(
            f"Upper-body joint-name count {len(upper_joint_names)} "
            f"does not match action dimension {expected_upper_dim}"
        )

    upper_scale_cfg = cfg.actions.upper_body_joint_pos.scale
    if not isinstance(upper_scale_cfg, float):
        raise TypeError("This test expects a scalar upper-body action scale")

    upper_scale = upper_scale_cfg

    active_action_indices = {name: upper_joint_names.index(name) for name in active_joint_names}

    active_joint_ids, resolved_active_names = robot.find_joints(active_joint_names)
    inactive_joint_ids, resolved_inactive_names = robot.find_joints(inactive_joint_names)
    all_arm_joint_ids, _ = robot.find_joints(LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES)

    frame_names = hand_frames.data.target_frame_names
    active_frame_id = frame_names.index(f"{active_side}_hand_palm")
    inactive_frame_id = frame_names.index(f"{inactive_side}_hand_palm")

    actions = torch.zeros(
        (1, env.action_manager.total_action_dim),
        device=env.device,
    )

    base_command = config["base_command"]
    lower_command = torch.tensor(
        [
            base_command["lin_vel_x"],
            base_command["lin_vel_y"],
            base_command["ang_vel_z"],
            base_command["base_height"],
        ],
        device=env.device,
    )

    def step_environment(upper_raw_action: torch.Tensor) -> None:
        actions.zero_()
        actions[:, upper_slice] = upper_raw_action
        actions[:, lower_slice] = lower_command
        env.step(actions)

    zero_upper_action = torch.zeros(
        expected_upper_dim,
        device=env.device,
    )

    warmup_steps = int(config["warmup_steps"])
    motion_steps = int(config["motion_steps"])
    settle_steps = int(config["settle_steps"])

    for step in range(warmup_steps):
        step_environment(zero_upper_action)

        if not bool(torch.isfinite(robot.data.root_state_w).all()):
            raise RuntimeError(f"Non-finite root state during warmup at step {step}")

    baseline_root_position = robot.data.root_pos_w[0].clone()
    baseline_root_quaternion = robot.data.root_quat_w[0].clone()

    baseline_palm_positions = hand_frames.data.target_pos_source[0].clone()
    baseline_active_palm = baseline_palm_positions[active_frame_id]
    baseline_inactive_palm = baseline_palm_positions[inactive_frame_id]

    baseline_inactive_joint_pos = robot.data.joint_pos[0, inactive_joint_ids].clone()

    requested_offsets = config["joint_offsets_rad"]

    joint_amplitudes: dict[str, float] = {}
    for suffix, amplitude in requested_offsets.items():
        full_name = f"{active_side}_{suffix}"
        if full_name not in active_action_indices:
            raise KeyError(f"Joint not present in upper action: {full_name}")
        joint_amplitudes[full_name] = float(amplitude)

    previous_offsets = dict.fromkeys(joint_amplitudes, 0.0)

    active_palm_positions: list[torch.Tensor] = []
    inactive_palm_positions: list[torch.Tensor] = []
    root_positions: list[torch.Tensor] = []
    root_quaternions: list[torch.Tensor] = []
    active_target_errors: list[torch.Tensor] = []
    inactive_joint_deviations: list[torch.Tensor] = []
    arm_torque_ratios: list[torch.Tensor] = []

    effort_limits = robot.data.joint_effort_limits[0, all_arm_joint_ids].abs().clamp_min(1.0e-6)

    for step in range(motion_steps):
        phase = (step + 1) / motion_steps

        # Starts at zero, reaches one halfway, and returns to zero.
        bump = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))

        upper_raw_action = torch.zeros(
            expected_upper_dim,
            device=env.device,
        )

        for joint_name, amplitude in joint_amplitudes.items():
            desired_offset = amplitude * bump
            offset_delta = desired_offset - previous_offsets[joint_name]

            action_index = active_action_indices[joint_name]
            upper_raw_action[action_index] = offset_delta / upper_scale

            previous_offsets[joint_name] = desired_offset

        step_environment(upper_raw_action)

        tensors = (
            robot.data.root_state_w,
            robot.data.joint_pos,
            robot.data.joint_vel,
            robot.data.applied_torque,
            hand_frames.data.target_pos_source,
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise RuntimeError(f"Non-finite state at motion step {step}")

        palm_positions = hand_frames.data.target_pos_source[0]

        active_palm_positions.append(palm_positions[active_frame_id].clone())
        inactive_palm_positions.append(palm_positions[inactive_frame_id].clone())
        root_positions.append(robot.data.root_pos_w[0].clone())
        root_quaternions.append(robot.data.root_quat_w[0].clone())

        active_target_errors.append(
            (
                robot.data.joint_pos[0, active_joint_ids]
                - robot.data.joint_pos_target[0, active_joint_ids]
            )
            .abs()
            .clone()
        )

        inactive_joint_deviations.append(
            (robot.data.joint_pos[0, inactive_joint_ids] - baseline_inactive_joint_pos)
            .abs()
            .clone()
        )

        arm_torque_ratios.append(
            (robot.data.applied_torque[0, all_arm_joint_ids].abs() / effort_limits).clone()
        )

    for _ in range(settle_steps):
        step_environment(zero_upper_action)

    final_palm_positions = hand_frames.data.target_pos_source[0]

    active_palm_positions_t = torch.stack(active_palm_positions)
    inactive_palm_positions_t = torch.stack(inactive_palm_positions)
    root_positions_t = torch.stack(root_positions)
    root_quaternions_t = torch.stack(root_quaternions)
    active_target_errors_t = torch.stack(active_target_errors)
    inactive_joint_deviations_t = torch.stack(inactive_joint_deviations)
    arm_torque_ratios_t = torch.stack(arm_torque_ratios)

    active_displacements = torch.linalg.vector_norm(
        active_palm_positions_t - baseline_active_palm,
        dim=-1,
    )
    inactive_displacements = torch.linalg.vector_norm(
        inactive_palm_positions_t - baseline_inactive_palm,
        dim=-1,
    )

    root_xy_excursions = torch.linalg.vector_norm(
        root_positions_t[:, :2] - baseline_root_position[:2],
        dim=-1,
    )

    root_orientation_errors = quaternion_error_deg(
        root_quaternions_t,
        baseline_root_quaternion,
    )

    active_return_error = float(
        torch.linalg.vector_norm(
            final_palm_positions[active_frame_id] - baseline_active_palm
        ).item()
    )

    metrics = {
        "test_id": config["test_id"],
        "active_side": active_side,
        "seed": cfg.seed,
        "step_dt_s": env.step_dt,
        "motion_duration_s": motion_steps * env.step_dt,
        "upper_joint_names": upper_joint_names,
        "resolved_active_joint_names": resolved_active_names,
        "resolved_inactive_joint_names": resolved_inactive_names,
        "active_hand_peak_displacement_m": float(active_displacements.max().item()),
        "active_hand_return_error_m": active_return_error,
        "inactive_hand_position_p95_m": p95(inactive_displacements),
        "inactive_hand_position_max_m": float(inactive_displacements.max().item()),
        "inactive_arm_joint_p95_deviation_rad": p95(inactive_joint_deviations_t),
        "active_arm_target_p95_error_rad": p95(active_target_errors_t),
        "active_arm_target_max_error_rad": float(active_target_errors_t.max().item()),
        "base_xy_excursion_max_m": float(root_xy_excursions.max().item()),
        "root_orientation_error_max_deg": float(root_orientation_errors.max().item()),
        "root_height_min_m": float(root_positions_t[:, 2].min().item()),
        "root_height_max_m": float(root_positions_t[:, 2].max().item()),
        "arm_torque_ratio_max": float(arm_torque_ratios_t.max().item()),
    }

    acceptance = config["acceptance"]

    checks = {
        "active_hand_moved": (
            metrics["active_hand_peak_displacement_m"]
            >= acceptance["minimum_active_hand_peak_displacement_m"]
        ),
        "inactive_hand_stable": (
            metrics["inactive_hand_position_p95_m"]
            <= acceptance["maximum_inactive_hand_position_p95_m"]
        ),
        "inactive_arm_stable": (
            metrics["inactive_arm_joint_p95_deviation_rad"]
            <= acceptance["maximum_inactive_arm_joint_p95_deviation_rad"]
        ),
        "active_hand_returned": (
            metrics["active_hand_return_error_m"]
            <= acceptance["maximum_active_hand_return_error_m"]
        ),
        "active_arm_tracking": (
            metrics["active_arm_target_p95_error_rad"]
            <= acceptance["maximum_active_arm_target_p95_error_rad"]
        ),
        "base_stable": (
            metrics["base_xy_excursion_max_m"] <= acceptance["maximum_base_xy_excursion_m"]
        ),
        "root_orientation_stable": (
            metrics["root_orientation_error_max_deg"]
            <= acceptance["maximum_root_orientation_error_deg"]
        ),
        "root_height_valid": (
            metrics["root_height_min_m"] > acceptance["minimum_root_height_m"]
            and metrics["root_height_max_m"] < acceptance["maximum_root_height_m"]
        ),
        "torque_valid": (metrics["arm_torque_ratio_max"] <= acceptance["maximum_arm_torque_ratio"]),
    }

    passed = all(checks.values())

    result = {
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "acceptance": acceptance,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"STAGE1_SINGLE_ARM_MOTION: {'PASS' if passed else 'FAIL'} side={active_side}",
        flush=True,
    )

    env.close()

    if not passed:
        failed_checks = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Failed checks: {failed_checks}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
