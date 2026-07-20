"""Run deterministic synchronized dual-arm motion."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=Path,
    default=PROJECT_ROOT / "configs/stage1/s1_05_dual_arm_sync.yaml",
)
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
    acceptance = config["acceptance"]

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
    upper_dim = upper_slice.stop - upper_slice.start

    if len(upper_joint_names) != upper_dim:
        raise RuntimeError("Upper-body joint ordering mismatch")

    upper_scale = cfg.actions.upper_body_joint_pos.scale
    if not isinstance(upper_scale, float):
        raise TypeError("Expected scalar upper-body action scale")

    all_arm_names = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
    arm_joint_ids, resolved_arm_names = robot.find_joints(all_arm_names)

    frame_names = hand_frames.data.target_frame_names
    left_frame_id = frame_names.index("left_hand_palm")
    right_frame_id = frame_names.index("right_hand_palm")

    controlled_names = []
    amplitudes = {}

    for side in ("left", "right"):
        for suffix, amplitude in config["joint_offsets_rad"].items():
            full_name = f"{side}_{suffix}"
            if full_name not in upper_joint_names:
                raise KeyError(f"Missing upper-body joint: {full_name}")
            controlled_names.append(full_name)
            amplitudes[full_name] = float(amplitude)

    action_indices = {name: upper_joint_names.index(name) for name in controlled_names}

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

    zero_upper = torch.zeros(upper_dim, device=env.device)

    def step_env(upper_action: torch.Tensor) -> None:
        actions.zero_()
        actions[:, upper_slice] = upper_action
        actions[:, lower_slice] = lower_command
        env.step(actions)

    for step in range(int(config["warmup_steps"])):
        step_env(zero_upper)

        if not bool(torch.isfinite(robot.data.root_state_w).all()):
            raise RuntimeError(f"Non-finite state during warmup: {step}")

    baseline_root_pos = robot.data.root_pos_w[0].clone()
    baseline_root_quat = robot.data.root_quat_w[0].clone()

    baseline_palms = hand_frames.data.target_pos_source[0].clone()
    baseline_left = baseline_palms[left_frame_id]
    baseline_right = baseline_palms[right_frame_id]

    previous_offsets = dict.fromkeys(controlled_names, 0.0)

    left_positions = []
    right_positions = []
    root_positions = []
    root_quaternions = []
    arm_target_errors = []
    torque_ratios = []

    effort_limits = robot.data.joint_effort_limits[0, arm_joint_ids].abs().clamp_min(1.0e-6)

    motion_steps = int(config["motion_steps"])

    for step in range(motion_steps):
        phase = (step + 1) / motion_steps

        # 0 -> 1 -> 0 smooth synchronized motion.
        bump = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))

        upper_action = torch.zeros(upper_dim, device=env.device)

        for name in controlled_names:
            desired_offset = amplitudes[name] * bump
            offset_delta = desired_offset - previous_offsets[name]
            upper_action[action_indices[name]] = offset_delta / upper_scale
            previous_offsets[name] = desired_offset

        step_env(upper_action)

        tensors = (
            robot.data.root_state_w,
            robot.data.joint_pos,
            robot.data.joint_pos_target,
            robot.data.applied_torque,
            hand_frames.data.target_pos_source,
        )

        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError(f"Non-finite state at motion step {step}")

        palms = hand_frames.data.target_pos_source[0]

        left_positions.append(palms[left_frame_id].clone())
        right_positions.append(palms[right_frame_id].clone())
        root_positions.append(robot.data.root_pos_w[0].clone())
        root_quaternions.append(robot.data.root_quat_w[0].clone())

        arm_target_errors.append(
            (robot.data.joint_pos[0, arm_joint_ids] - robot.data.joint_pos_target[0, arm_joint_ids])
            .abs()
            .clone()
        )

        torque_ratios.append(
            (robot.data.applied_torque[0, arm_joint_ids].abs() / effort_limits).clone()
        )

    for _ in range(int(config["settle_steps"])):
        step_env(zero_upper)

    final_palms = hand_frames.data.target_pos_source[0]

    left_positions_t = torch.stack(left_positions)
    right_positions_t = torch.stack(right_positions)
    root_positions_t = torch.stack(root_positions)
    root_quaternions_t = torch.stack(root_quaternions)
    arm_target_errors_t = torch.stack(arm_target_errors)
    torque_ratios_t = torch.stack(torque_ratios)

    left_displacements = torch.linalg.vector_norm(
        left_positions_t - baseline_left,
        dim=-1,
    )
    right_displacements = torch.linalg.vector_norm(
        right_positions_t - baseline_right,
        dim=-1,
    )

    left_peak = float(left_displacements.max().item())
    right_peak = float(right_displacements.max().item())

    left_peak_step = int(torch.argmax(left_displacements).item())
    right_peak_step = int(torch.argmax(right_displacements).item())

    peak_time_difference = abs(left_peak_step - right_peak_step) * env.step_dt

    trace_difference = (left_displacements - right_displacements).abs()

    root_xy_excursions = torch.linalg.vector_norm(
        root_positions_t[:, :2] - baseline_root_pos[:2],
        dim=-1,
    )

    root_orientation_errors = quaternion_error_deg(
        root_quaternions_t,
        baseline_root_quat,
    )

    left_return_error = float(
        torch.linalg.vector_norm(final_palms[left_frame_id] - baseline_left).item()
    )

    right_return_error = float(
        torch.linalg.vector_norm(final_palms[right_frame_id] - baseline_right).item()
    )

    metrics = {
        "test_id": config["test_id"],
        "seed": cfg.seed,
        "step_dt_s": env.step_dt,
        "motion_duration_s": motion_steps * env.step_dt,
        "resolved_arm_joint_names": resolved_arm_names,
        "left_hand_peak_displacement_m": left_peak,
        "right_hand_peak_displacement_m": right_peak,
        "peak_displacement_difference_m": abs(left_peak - right_peak),
        "left_peak_step": left_peak_step,
        "right_peak_step": right_peak_step,
        "peak_time_difference_s": peak_time_difference,
        "displacement_trace_p95_difference_m": p95(trace_difference),
        "left_hand_return_error_m": left_return_error,
        "right_hand_return_error_m": right_return_error,
        "arm_target_error_p95_rad": p95(arm_target_errors_t),
        "arm_target_error_max_rad": float(arm_target_errors_t.max().item()),
        "base_xy_excursion_max_m": float(root_xy_excursions.max().item()),
        "root_orientation_error_max_deg": float(root_orientation_errors.max().item()),
        "root_height_min_m": float(root_positions_t[:, 2].min().item()),
        "root_height_max_m": float(root_positions_t[:, 2].max().item()),
        "arm_torque_ratio_max": float(torque_ratios_t.max().item()),
    }

    checks = {
        "left_hand_moved": (
            metrics["left_hand_peak_displacement_m"]
            >= acceptance["minimum_left_hand_peak_displacement_m"]
        ),
        "right_hand_moved": (
            metrics["right_hand_peak_displacement_m"]
            >= acceptance["minimum_right_hand_peak_displacement_m"]
        ),
        "peak_displacements_match": (
            metrics["peak_displacement_difference_m"]
            <= acceptance["maximum_peak_displacement_difference_m"]
        ),
        "peak_times_match": (
            metrics["peak_time_difference_s"] <= acceptance["maximum_peak_time_difference_s"]
        ),
        "motion_traces_match": (
            metrics["displacement_trace_p95_difference_m"]
            <= acceptance["maximum_displacement_trace_p95_difference_m"]
        ),
        "left_hand_returned": (
            metrics["left_hand_return_error_m"] <= acceptance["maximum_left_hand_return_error_m"]
        ),
        "right_hand_returned": (
            metrics["right_hand_return_error_m"] <= acceptance["maximum_right_hand_return_error_m"]
        ),
        "arm_tracking": (
            metrics["arm_target_error_p95_rad"] <= acceptance["maximum_arm_target_p95_error_rad"]
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
        "STAGE1_S1_05_DUAL_ARM_SYNC: " + ("PASS" if passed else "FAIL"),
        flush=True,
    )

    env.close()

    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Failed checks: {failed}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
