"""Run the Stage-1 60-second fixed bimanual hold regression."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=Path,
    default=PROJECT_ROOT / "configs/stage1/s1_02_dual_arm_hold.yaml",
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
    """Return the slice occupied by one action term."""

    index = names.index(name)
    start = sum(dims[:index])
    return slice(start, start + dims[index])


def rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values.square())).item())


def p95(values: torch.Tensor) -> float:
    return float(torch.quantile(values.flatten(), 0.95).item())


def quaternion_error_deg(
    quaternions: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return shortest quaternion rotation error in degrees."""

    dots = torch.sum(quaternions * reference, dim=-1).abs()
    dots = torch.clamp(dots, 0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dots))


def main() -> None:
    config = yaml.safe_load(args.config.read_text())

    warmup_steps = int(config["warmup_steps"])
    measurement_steps = int(config["measurement_steps"])
    command = config["command"]
    acceptance = config["acceptance"]

    cfg = G1Stage1NoBoxEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = int(config["seed"])
    cfg.sim.device = args.device

    env = ManagerBasedEnv(cfg=cfg)
    env.reset(seed=cfg.seed)

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

    actions = torch.zeros(
        (1, env.action_manager.total_action_dim),
        device=env.device,
    )

    # Upper-body zero delta means: keep the current fixed joint targets.
    actions[:, upper_slice] = 0.0

    # AGILE lower-body input: [vx, vy, wz, pelvis height].
    actions[:, lower_slice] = torch.tensor(
        [
            command["lin_vel_x"],
            command["lin_vel_y"],
            command["ang_vel_z"],
            command["base_height"],
        ],
        device=env.device,
    )

    robot = env.scene["robot"]
    hand_frames = env.scene["hand_frames"]

    arm_joint_ids, arm_joint_names = robot.find_joints(LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES)

    frame_names = hand_frames.data.target_frame_names
    left_frame_id = frame_names.index("left_hand_palm")
    right_frame_id = frame_names.index("right_hand_palm")

    for step in range(warmup_steps):
        env.step(actions)

        if not bool(torch.isfinite(robot.data.root_state_w).all()):
            raise RuntimeError(f"Non-finite root state during warmup at step {step}")

    root_positions = torch.empty(
        (measurement_steps, 3),
        device=env.device,
    )
    root_quaternions = torch.empty(
        (measurement_steps, 4),
        device=env.device,
    )
    palm_positions = torch.empty(
        (measurement_steps, 2, 3),
        device=env.device,
    )
    palm_quaternions = torch.empty(
        (measurement_steps, 2, 4),
        device=env.device,
    )
    arm_positions = torch.empty(
        (measurement_steps, len(arm_joint_ids)),
        device=env.device,
    )
    arm_targets = torch.empty_like(arm_positions)
    arm_velocities = torch.empty_like(arm_positions)
    arm_torque_ratios = torch.empty_like(arm_positions)

    effort_limits = robot.data.joint_effort_limits[0, arm_joint_ids].abs().clamp_min(1.0e-6)

    all_numeric_finite = True

    for step in range(measurement_steps):
        env.step(actions)

        root_state = robot.data.root_state_w[0]
        joint_pos = robot.data.joint_pos[0, arm_joint_ids]
        joint_target = robot.data.joint_pos_target[0, arm_joint_ids]
        joint_vel = robot.data.joint_vel[0, arm_joint_ids]
        applied_torque = robot.data.applied_torque[0, arm_joint_ids]

        current_palm_positions = hand_frames.data.target_pos_source[0]
        current_palm_quaternions = hand_frames.data.target_quat_source[0]

        tensors = (
            root_state,
            joint_pos,
            joint_target,
            joint_vel,
            applied_torque,
            current_palm_positions,
            current_palm_quaternions,
        )

        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            all_numeric_finite = False
            raise RuntimeError(f"Non-finite state at measurement step {step}")

        root_positions[step] = root_state[:3]
        root_quaternions[step] = root_state[3:7]

        palm_positions[step, 0] = current_palm_positions[left_frame_id]
        palm_positions[step, 1] = current_palm_positions[right_frame_id]
        palm_quaternions[step, 0] = current_palm_quaternions[left_frame_id]
        palm_quaternions[step, 1] = current_palm_quaternions[right_frame_id]

        arm_positions[step] = joint_pos
        arm_targets[step] = joint_target
        arm_velocities[step] = joint_vel
        arm_torque_ratios[step] = applied_torque.abs() / effort_limits

    step_dt = env.step_dt

    root_reference = root_positions[0]
    root_quaternion_reference = root_quaternions[0]

    xy_offsets = root_positions[:, :2] - root_reference[:2]
    xy_excursions = torch.linalg.vector_norm(xy_offsets, dim=-1)

    root_orientation_errors = quaternion_error_deg(
        root_quaternions,
        root_quaternion_reference,
    )

    base_height_errors = (root_positions[:, 2] - float(command["base_height"])).abs()

    palm_position_errors = torch.linalg.vector_norm(
        palm_positions - palm_positions[0:1],
        dim=-1,
    )

    left_orientation_errors = quaternion_error_deg(
        palm_quaternions[:, 0],
        palm_quaternions[0, 0],
    )
    right_orientation_errors = quaternion_error_deg(
        palm_quaternions[:, 1],
        palm_quaternions[0, 1],
    )

    arm_target_errors = (arm_positions - arm_targets).abs()

    arm_accelerations = (
        torch.diff(
            arm_velocities,
            dim=0,
        )
        / step_dt
    )

    arm_jerks = (
        torch.diff(
            arm_accelerations,
            dim=0,
        )
        / step_dt
    )

    joint_limits = robot.data.joint_pos_limits[0, arm_joint_ids]
    lower_margins = arm_positions - joint_limits[:, 0]
    upper_margins = joint_limits[:, 1] - arm_positions
    joint_limit_margin = torch.minimum(lower_margins, upper_margins)

    metrics = {
        "test_id": config["test_id"],
        "seed": cfg.seed,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "measurement_duration_s": measurement_steps * step_dt,
        "step_dt_s": step_dt,
        "all_numeric_finite": all_numeric_finite,
        "action_terms": action_names,
        "action_dims": action_dims,
        "arm_joint_names": arm_joint_names,
        "root_height_min_m": float(root_positions[:, 2].min().item()),
        "root_height_mean_m": float(root_positions[:, 2].mean().item()),
        "root_height_max_m": float(root_positions[:, 2].max().item()),
        "base_height_error_p95_m": p95(base_height_errors),
        "final_xy_drift_m": float(xy_excursions[-1].item()),
        "maximum_xy_excursion_m": float(xy_excursions.max().item()),
        "root_orientation_error_max_deg": float(root_orientation_errors.max().item()),
        "left_hand_position_rms_m": rms(palm_position_errors[:, 0]),
        "left_hand_position_p95_m": p95(palm_position_errors[:, 0]),
        "left_hand_position_max_m": float(palm_position_errors[:, 0].max().item()),
        "right_hand_position_rms_m": rms(palm_position_errors[:, 1]),
        "right_hand_position_p95_m": p95(palm_position_errors[:, 1]),
        "right_hand_position_max_m": float(palm_position_errors[:, 1].max().item()),
        "left_hand_orientation_rms_deg": rms(left_orientation_errors),
        "left_hand_orientation_p95_deg": p95(left_orientation_errors),
        "right_hand_orientation_rms_deg": rms(right_orientation_errors),
        "right_hand_orientation_p95_deg": p95(right_orientation_errors),
        "arm_joint_target_error_rms_rad": rms(arm_target_errors),
        "arm_joint_target_error_p95_rad": p95(arm_target_errors),
        "arm_joint_target_error_max_rad": float(arm_target_errors.max().item()),
        "arm_joint_velocity_p95_rad_s": p95(arm_velocities.abs()),
        "arm_joint_velocity_max_rad_s": float(arm_velocities.abs().max().item()),
        "arm_joint_acceleration_p95_rad_s2": p95(arm_accelerations.abs()),
        "arm_joint_acceleration_max_rad_s2": float(arm_accelerations.abs().max().item()),
        "arm_joint_jerk_p95_rad_s3": p95(arm_jerks.abs()),
        "arm_joint_jerk_max_rad_s3": float(arm_jerks.abs().max().item()),
        "arm_torque_ratio_p95": p95(arm_torque_ratios),
        "arm_torque_ratio_max": float(arm_torque_ratios.max().item()),
        "minimum_arm_joint_limit_margin_rad": float(joint_limit_margin.min().item()),
    }

    checks = {
        "numeric_finite": metrics["all_numeric_finite"],
        "root_height_min": (metrics["root_height_min_m"] > acceptance["minimum_root_height_m"]),
        "root_height_max": (metrics["root_height_max_m"] < acceptance["maximum_root_height_m"]),
        "base_height_tracking": (
            metrics["base_height_error_p95_m"] <= acceptance["maximum_base_height_p95_error_m"]
        ),
        "xy_drift": (metrics["final_xy_drift_m"] <= acceptance["maximum_final_xy_drift_m"]),
        "root_orientation": (
            metrics["root_orientation_error_max_deg"]
            <= acceptance["maximum_root_orientation_error_deg"]
        ),
        "left_hand_position": (
            metrics["left_hand_position_rms_m"] <= acceptance["maximum_hand_position_rms_m"]
            and metrics["left_hand_position_p95_m"] <= acceptance["maximum_hand_position_p95_m"]
        ),
        "right_hand_position": (
            metrics["right_hand_position_rms_m"] <= acceptance["maximum_hand_position_rms_m"]
            and metrics["right_hand_position_p95_m"] <= acceptance["maximum_hand_position_p95_m"]
        ),
        "left_hand_orientation": (
            metrics["left_hand_orientation_rms_deg"]
            <= acceptance["maximum_hand_orientation_rms_deg"]
            and metrics["left_hand_orientation_p95_deg"]
            <= acceptance["maximum_hand_orientation_p95_deg"]
        ),
        "right_hand_orientation": (
            metrics["right_hand_orientation_rms_deg"]
            <= acceptance["maximum_hand_orientation_rms_deg"]
            and metrics["right_hand_orientation_p95_deg"]
            <= acceptance["maximum_hand_orientation_p95_deg"]
        ),
        "arm_joint_target": (
            metrics["arm_joint_target_error_p95_rad"]
            <= acceptance["maximum_arm_joint_target_p95_error_rad"]
        ),
        "torque_limit": (metrics["arm_torque_ratio_max"] <= acceptance["maximum_torque_ratio"]),
        "joint_limits": (
            metrics["minimum_arm_joint_limit_margin_rad"]
            >= acceptance["minimum_joint_limit_margin_rad"]
        ),
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
        "STAGE1_S1_02_DUAL_ARM_HOLD: " + ("PASS" if passed else "FAIL"),
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
