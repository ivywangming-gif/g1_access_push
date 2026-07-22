"""Validate one isolated virtual-box small-arc trajectory."""

from __future__ import annotations

import argparse
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
import math  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402

from g1_access_push.sim.stage1.no_box_env_cfg import (  # noqa: E402
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)
from g1_access_push.sim.stage1.virtual_box_env_cfg import (  # noqa: E402
    G1Stage1VirtualBoxRecurrentEnvCfg,
)
from g1_access_push.sim.stage1.virtual_box_trajectory import (  # noqa: E402
    derive_equal_budget_small_arc,
    straight_out_hold_back_displacement,
    virtual_surface_arc_targets,
)


def term_slice(
    names: list[str],
    dims: list[int],
    name: str,
) -> slice:
    index = names.index(name)
    start = sum(dims[:index])
    return slice(start, start + dims[index])


def p95(values: torch.Tensor) -> float:
    return float(torch.quantile(values.flatten(), 0.95).item())


def quaternion_error_deg(
    quaternions: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    dots = torch.sum(
        quaternions * reference,
        dim=-1,
    ).abs()
    dots = torch.clamp(dots, 0.0, 1.0)

    return torch.rad2deg(2.0 * torch.acos(dots))


def root_tilt_deg(
    quaternions: torch.Tensor,
) -> torch.Tensor:
    x = quaternions[..., 1]
    y = quaternions[..., 2]

    cosine = 1.0 - 2.0 * (x.square() + y.square())
    cosine = torch.clamp(cosine, -1.0, 1.0)

    return torch.rad2deg(torch.acos(cosine))


def clamp_vector_norm(
    vectors: torch.Tensor,
    maximum_norm: float,
) -> torch.Tensor:
    norms = torch.linalg.vector_norm(
        vectors,
        dim=-1,
        keepdim=True,
    )
    scales = torch.clamp(
        maximum_norm / norms.clamp_min(1.0e-9),
        max=1.0,
    )

    return vectors * scales


def wrapped_angle_difference(
    angles: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    difference = angles - reference

    return torch.atan2(
        torch.sin(difference),
        torch.cos(difference),
    )


def planar_pair_yaw(
    separation_vectors: torch.Tensor,
    baseline_separation: torch.Tensor,
) -> torch.Tensor:
    vectors_xy = separation_vectors[..., :2]
    reference_xy = baseline_separation[:2]

    cross = reference_xy[0] * vectors_xy[..., 1] - reference_xy[1] * vectors_xy[..., 0]
    dot = torch.sum(
        vectors_xy * reference_xy.unsqueeze(0),
        dim=-1,
    )

    return torch.atan2(cross, dot)


def main() -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if config.get("episode_mode") != "isolated":
        raise ValueError("S1-07B requires episode_mode='isolated'.")

    case = config["case"]
    trajectory = config["trajectory"]
    acceptance = config["acceptance"]

    case_name = str(case["name"])
    direction = int(case["direction"])

    if direction not in (-1, 1):
        raise ValueError("Arc direction must be -1 or +1.")

    cfg = G1Stage1VirtualBoxRecurrentEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = int(config["seed"])
    cfg.sim.device = args.device

    env = ManagerBasedEnv(cfg=cfg)

    try:
        env.reset(seed=cfg.seed)

        robot = env.scene["robot"]
        hand_frames = env.scene["hand_frames"]

        action_names = env.action_manager.active_terms
        action_dims = env.action_manager.action_term_dim

        resolved_action_dims = {
            name: int(dim)
            for name, dim in zip(
                action_names,
                action_dims,
                strict=True,
            )
        }

        expected_action_dims = {
            "left_hand_pose": 6,
            "right_hand_pose": 6,
            "waist_joint_pos": 3,
            "lower_body_joint_pos": 4,
        }

        if resolved_action_dims != expected_action_dims:
            raise RuntimeError(f"Unexpected action contract: {resolved_action_dims}")

        left_slice = term_slice(
            action_names,
            action_dims,
            "left_hand_pose",
        )
        right_slice = term_slice(
            action_names,
            action_dims,
            "right_hand_pose",
        )
        waist_slice = term_slice(
            action_names,
            action_dims,
            "waist_joint_pos",
        )
        lower_slice = term_slice(
            action_names,
            action_dims,
            "lower_body_joint_pos",
        )

        lower_term = env.action_manager.get_term("lower_body_joint_pos")

        frame_names = hand_frames.data.target_frame_names
        left_frame_id = frame_names.index("left_hand_palm")
        right_frame_id = frame_names.index("right_hand_palm")

        arm_joint_ids, resolved_arm_names = robot.find_joints(
            LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
        )

        actions = torch.zeros(
            (
                1,
                env.action_manager.total_action_dim,
            ),
            dtype=torch.float32,
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
            dtype=torch.float32,
            device=env.device,
        )

        def step_zero_pose() -> None:
            actions.zero_()
            actions[:, waist_slice] = 0.0
            actions[:, lower_slice] = lower_command
            env.step(actions)

        for step in range(int(config["warmup_steps"])):
            step_zero_pose()

            tensors = (
                robot.data.root_state_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
                lower_term.policy_actions,
                lower_term.last_policy_input,
                hand_frames.data.target_pos_source,
                hand_frames.data.target_quat_source,
            )

            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                raise RuntimeError(f"Non-finite warmup state at step {step}.")

        baseline_root_pos = robot.data.root_pos_w[0].clone()
        baseline_root_quat = robot.data.root_quat_w[0].clone()

        baseline_positions = hand_frames.data.target_pos_source[0].clone()
        baseline_quaternions = hand_frames.data.target_quat_source[0].clone()

        baseline_midpoint = baseline_positions.mean(dim=0)
        baseline_separation = baseline_positions[right_frame_id] - baseline_positions[left_frame_id]
        baseline_separation_m = float(torch.linalg.vector_norm(baseline_separation).item())

        arc_geometry = derive_equal_budget_small_arc(
            contact_separation_m=(baseline_separation_m),
            local_motion_budget_m=float(trajectory["local_motion_budget_m"]),
        )

        if (
            arc_geometry["conservative_contact_displacement_bound_m"]
            > float(trajectory["local_motion_budget_m"]) + 1.0e-9
        ):
            raise RuntimeError("Derived arc exceeds the validated local motion budget.")

        effort_limits = (
            robot.data.joint_effort_limits[
                0,
                arm_joint_ids,
            ]
            .abs()
            .clamp_min(1.0e-6)
        )
        joint_limits = robot.data.joint_pos_limits[
            0,
            arm_joint_ids,
        ]

        move_out_steps = int(trajectory["move_out_steps"])
        hold_steps = int(trajectory["hold_steps"])
        move_back_steps = int(trajectory["move_back_steps"])
        settle_steps = int(trajectory["settle_steps"])
        transient_ignore = int(config["transient_ignore_steps"])

        if hold_steps <= transient_ignore:
            raise ValueError("hold_steps must exceed transient ignore.")

        if settle_steps <= transient_ignore:
            raise ValueError("settle_steps must exceed transient ignore.")

        profile_steps = move_out_steps + hold_steps + move_back_steps
        total_steps = profile_steps + settle_steps

        actual_positions: list[torch.Tensor] = []
        actual_quaternions: list[torch.Tensor] = []
        desired_positions_log: list[torch.Tensor] = []
        desired_quaternions_log: list[torch.Tensor] = []
        desired_midpoints: list[torch.Tensor] = []
        desired_yaws: list[torch.Tensor] = []
        root_positions: list[torch.Tensor] = []
        root_quaternions: list[torch.Tensor] = []
        root_tilts: list[torch.Tensor] = []
        arm_target_errors: list[torch.Tensor] = []
        arm_velocities: list[torch.Tensor] = []
        torque_ratios: list[torch.Tensor] = []
        joint_limit_margins: list[torch.Tensor] = []

        maximum_position_correction = float(trajectory["maximum_position_correction_m"])
        maximum_orientation_correction = float(trajectory["maximum_orientation_correction_rad"])

        all_numeric_finite = True

        for step in range(total_steps):
            if step < profile_steps:
                fraction = straight_out_hold_back_displacement(
                    step,
                    amplitude_m=1.0,
                    move_out_steps=move_out_steps,
                    hold_steps=hold_steps,
                    move_back_steps=move_back_steps,
                )
            else:
                fraction = 0.0

            (
                desired_positions,
                desired_quaternions,
                midpoint_translation,
                desired_yaw,
            ) = virtual_surface_arc_targets(
                baseline_positions,
                baseline_quaternions,
                fraction=fraction,
                direction=direction,
                center_radius_m=arc_geometry["center_radius_m"],
                peak_yaw_rad=arc_geometry["peak_yaw_rad"],
            )

            current_positions = hand_frames.data.target_pos_source[0]
            current_quaternions = hand_frames.data.target_quat_source[0]

            (
                raw_position_error,
                raw_orientation_error,
            ) = math_utils.compute_pose_error(
                current_positions,
                current_quaternions,
                desired_positions,
                desired_quaternions,
                rot_error_type="axis_angle",
            )

            position_correction = clamp_vector_norm(
                raw_position_error,
                maximum_position_correction,
            )
            orientation_correction = clamp_vector_norm(
                raw_orientation_error,
                maximum_orientation_correction,
            )

            actions.zero_()
            actions[:, left_slice] = torch.cat(
                (
                    position_correction[left_frame_id],
                    orientation_correction[left_frame_id],
                )
            )
            actions[:, right_slice] = torch.cat(
                (
                    position_correction[right_frame_id],
                    orientation_correction[right_frame_id],
                )
            )
            actions[:, waist_slice] = 0.0
            actions[:, lower_slice] = lower_command

            env.step(actions)

            actual_position = hand_frames.data.target_pos_source[0]
            actual_quaternion = hand_frames.data.target_quat_source[0]

            current_arm_positions = robot.data.joint_pos[
                0,
                arm_joint_ids,
            ]
            current_arm_targets = robot.data.joint_pos_target[
                0,
                arm_joint_ids,
            ]

            lower_margin = current_arm_positions - joint_limits[:, 0]
            upper_margin = joint_limits[:, 1] - current_arm_positions

            tensors = (
                robot.data.root_state_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
                robot.data.applied_torque,
                actual_position,
                actual_quaternion,
                desired_positions,
                desired_quaternions,
                lower_term.policy_actions,
                lower_term.processed_actions,
                lower_term.last_policy_input,
            )

            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                all_numeric_finite = False
                raise RuntimeError(f"Non-finite state in {case_name}, step {step}.")

            actual_positions.append(actual_position.clone())
            actual_quaternions.append(actual_quaternion.clone())
            desired_positions_log.append(desired_positions.clone())
            desired_quaternions_log.append(desired_quaternions.clone())
            desired_midpoints.append((baseline_midpoint + midpoint_translation).clone())
            desired_yaws.append(
                torch.tensor(
                    desired_yaw,
                    device=env.device,
                )
            )
            root_positions.append(robot.data.root_pos_w[0].clone())
            root_quaternions.append(robot.data.root_quat_w[0].clone())
            root_tilts.append(root_tilt_deg(robot.data.root_quat_w[0].unsqueeze(0))[0])
            arm_target_errors.append((current_arm_positions - current_arm_targets).abs().clone())
            arm_velocities.append(
                robot.data.joint_vel[
                    0,
                    arm_joint_ids,
                ].clone()
            )
            torque_ratios.append(
                (
                    robot.data.applied_torque[
                        0,
                        arm_joint_ids,
                    ].abs()
                    / effort_limits
                ).clone()
            )
            joint_limit_margins.append(
                torch.minimum(
                    lower_margin,
                    upper_margin,
                ).clone()
            )

        actual_positions_t = torch.stack(actual_positions)
        actual_quaternions_t = torch.stack(actual_quaternions)
        desired_positions_t = torch.stack(desired_positions_log)
        desired_quaternions_t = torch.stack(desired_quaternions_log)
        desired_midpoints_t = torch.stack(desired_midpoints)
        desired_yaws_t = torch.stack(desired_yaws)
        root_positions_t = torch.stack(root_positions)
        root_quaternions_t = torch.stack(root_quaternions)
        root_tilts_t = torch.stack(root_tilts)
        arm_target_errors_t = torch.stack(arm_target_errors)
        arm_velocities_t = torch.stack(arm_velocities)
        torque_ratios_t = torch.stack(torque_ratios)
        joint_limit_margins_t = torch.stack(joint_limit_margins)

        hand_position_errors = torch.linalg.vector_norm(
            actual_positions_t - desired_positions_t,
            dim=-1,
        )
        hand_orientation_errors = quaternion_error_deg(
            actual_quaternions_t,
            desired_quaternions_t,
        )

        actual_midpoints = actual_positions_t.mean(dim=1)
        midpoint_errors = torch.linalg.vector_norm(
            actual_midpoints - desired_midpoints_t,
            dim=-1,
        )

        actual_separation_vectors = (
            actual_positions_t[
                :,
                right_frame_id,
            ]
            - actual_positions_t[
                :,
                left_frame_id,
            ]
        )
        desired_separation_vectors = (
            desired_positions_t[
                :,
                right_frame_id,
            ]
            - desired_positions_t[
                :,
                left_frame_id,
            ]
        )

        separation_vector_errors = torch.linalg.vector_norm(
            actual_separation_vectors - desired_separation_vectors,
            dim=-1,
        )
        separation_magnitude_errors = (
            torch.linalg.vector_norm(
                actual_separation_vectors,
                dim=-1,
            )
            - baseline_separation_m
        ).abs()

        actual_pair_yaws = planar_pair_yaw(
            actual_separation_vectors,
            baseline_separation,
        )

        hold_start = move_out_steps + transient_ignore
        hold_end = move_out_steps + hold_steps

        settle_start = profile_steps + transient_ignore
        settle_end = total_steps

        steady_actual_yaws = actual_pair_yaws[hold_start:hold_end]
        steady_desired_yaws = desired_yaws_t[hold_start:hold_end]

        steady_yaw_errors = wrapped_angle_difference(
            steady_actual_yaws,
            steady_desired_yaws,
        ).abs()

        median_actual_yaw = float(torch.median(steady_actual_yaws).item())
        target_peak_yaw = float(direction) * arc_geometry["peak_yaw_rad"]

        vector_budget = float(acceptance["maximum_separation_vector_p95_error_m"])

        yaw_tolerance_from_vector_budget = 2.0 * math.asin(
            min(
                1.0,
                vector_budget / (2.0 * baseline_separation_m),
            )
        )

        minimum_yaw_response = max(
            0.0,
            abs(target_peak_yaw) - yaw_tolerance_from_vector_budget,
        )

        left_return = torch.linalg.vector_norm(
            actual_positions_t[
                settle_start:settle_end,
                left_frame_id,
            ]
            - baseline_positions[left_frame_id],
            dim=-1,
        )
        right_return = torch.linalg.vector_norm(
            actual_positions_t[
                settle_start:settle_end,
                right_frame_id,
            ]
            - baseline_positions[right_frame_id],
            dim=-1,
        )

        root_xy_excursions = torch.linalg.vector_norm(
            root_positions_t[:, :2] - baseline_root_pos[:2].unsqueeze(0),
            dim=-1,
        )

        root_orientation_errors = quaternion_error_deg(
            root_quaternions_t,
            baseline_root_quat,
        )

        arm_accelerations = (
            torch.diff(
                arm_velocities_t,
                dim=0,
            )
            / env.step_dt
        )
        arm_jerks = (
            torch.diff(
                arm_accelerations,
                dim=0,
            )
            / env.step_dt
        )

        metrics = {
            "test_id": config["test_id"],
            "case_name": case_name,
            "direction": direction,
            "seed": cfg.seed,
            "step_dt_s": env.step_dt,
            "action_terms": action_names,
            "action_dims": action_dims,
            "resolved_arm_joint_names": (resolved_arm_names),
            "baseline_contact_separation_m": (baseline_separation_m),
            "arc_geometry": arc_geometry,
            "target_peak_yaw_rad": (target_peak_yaw),
            "target_peak_yaw_deg": (math.degrees(target_peak_yaw)),
            "median_actual_steady_yaw_rad": (median_actual_yaw),
            "median_actual_steady_yaw_deg": (math.degrees(median_actual_yaw)),
            "yaw_error_p95_deg": (math.degrees(p95(steady_yaw_errors))),
            "derived_yaw_tolerance_deg": (math.degrees(yaw_tolerance_from_vector_budget)),
            "derived_minimum_yaw_response_deg": (math.degrees(minimum_yaw_response)),
            "arc_midpoint_position_p95_error_m": p95(midpoint_errors[hold_start:hold_end]),
            "arc_midpoint_position_max_error_m": float(midpoint_errors.max().item()),
            "separation_vector_p95_error_m": p95(separation_vector_errors),
            "separation_vector_max_error_m": float(separation_vector_errors.max().item()),
            "contact_separation_p95_error_m": p95(separation_magnitude_errors),
            "contact_separation_max_error_m": float(separation_magnitude_errors.max().item()),
            "left_hand_position_p95_m": p95(
                hand_position_errors[
                    :,
                    left_frame_id,
                ]
            ),
            "left_hand_position_max_m": float(
                hand_position_errors[
                    :,
                    left_frame_id,
                ]
                .max()
                .item()
            ),
            "right_hand_position_p95_m": p95(
                hand_position_errors[
                    :,
                    right_frame_id,
                ]
            ),
            "right_hand_position_max_m": float(
                hand_position_errors[
                    :,
                    right_frame_id,
                ]
                .max()
                .item()
            ),
            "left_hand_orientation_p95_deg": p95(
                hand_orientation_errors[
                    :,
                    left_frame_id,
                ]
            ),
            "left_hand_orientation_max_deg": float(
                hand_orientation_errors[
                    :,
                    left_frame_id,
                ]
                .max()
                .item()
            ),
            "right_hand_orientation_p95_deg": p95(
                hand_orientation_errors[
                    :,
                    right_frame_id,
                ]
            ),
            "right_hand_orientation_max_deg": float(
                hand_orientation_errors[
                    :,
                    right_frame_id,
                ]
                .max()
                .item()
            ),
            "left_hand_return_p95_m": p95(left_return),
            "left_hand_return_max_m": float(left_return.max().item()),
            "right_hand_return_p95_m": p95(right_return),
            "right_hand_return_max_m": float(right_return.max().item()),
            "arm_joint_target_p95_error_rad": p95(arm_target_errors_t),
            "arm_joint_target_max_error_rad": float(arm_target_errors_t.max().item()),
            "arm_joint_velocity_p95_rad_s": p95(arm_velocities_t.abs()),
            "arm_joint_acceleration_p95_rad_s2": p95(arm_accelerations.abs()),
            "arm_joint_jerk_p95_rad_s3": p95(arm_jerks.abs()),
            "arm_torque_ratio_p95": p95(torque_ratios_t),
            "arm_torque_ratio_max": float(torque_ratios_t.max().item()),
            "arm_torque_near_limit_fraction": float(
                (torque_ratios_t >= 0.99).float().mean().item()
            ),
            "minimum_arm_joint_limit_margin_rad": float(joint_limit_margins_t.min().item()),
            "base_xy_excursion_max_m": float(root_xy_excursions.max().item()),
            "root_orientation_error_max_deg": float(root_orientation_errors.max().item()),
            "root_tilt_max_deg": float(root_tilts_t.max().item()),
            "root_height_min_m": float(root_positions_t[:, 2].min().item()),
            "root_height_max_m": float(root_positions_t[:, 2].max().item()),
            "hidden_state_max": (lower_term.recurrent_state_max),
            "cell_state_max": (lower_term.recurrent_cell_max),
            "all_numeric_finite": (all_numeric_finite),
        }

        checks = {
            "action_contract": (resolved_action_dims == expected_action_dims),
            "finite": all_numeric_finite,
            "arc_midpoint_tracking": (
                metrics["arc_midpoint_position_p95_error_m"]
                <= acceptance["maximum_arc_midpoint_position_p95_error_m"]
                and metrics["arc_midpoint_position_max_error_m"]
                <= acceptance["maximum_arc_midpoint_position_max_error_m"]
            ),
            "arc_yaw_response": (
                median_actual_yaw * direction > 0.0
                and abs(median_actual_yaw) >= minimum_yaw_response
            ),
            "separation_vector_tracking": (
                metrics["separation_vector_p95_error_m"]
                <= acceptance["maximum_separation_vector_p95_error_m"]
                and metrics["separation_vector_max_error_m"]
                <= acceptance["maximum_separation_vector_max_error_m"]
            ),
            "contact_separation": (
                metrics["contact_separation_p95_error_m"]
                <= acceptance["maximum_contact_separation_p95_error_m"]
                and metrics["contact_separation_max_error_m"]
                <= acceptance["maximum_contact_separation_max_error_m"]
            ),
            "left_hand_position": (
                metrics["left_hand_position_p95_m"] <= acceptance["maximum_hand_position_p95_m"]
                and metrics["left_hand_position_max_m"] <= acceptance["maximum_hand_position_max_m"]
            ),
            "right_hand_position": (
                metrics["right_hand_position_p95_m"] <= acceptance["maximum_hand_position_p95_m"]
                and metrics["right_hand_position_max_m"]
                <= acceptance["maximum_hand_position_max_m"]
            ),
            "left_hand_orientation": (
                metrics["left_hand_orientation_p95_deg"]
                <= acceptance["maximum_hand_orientation_p95_deg"]
                and metrics["left_hand_orientation_max_deg"]
                <= acceptance["maximum_hand_orientation_max_deg"]
            ),
            "right_hand_orientation": (
                metrics["right_hand_orientation_p95_deg"]
                <= acceptance["maximum_hand_orientation_p95_deg"]
                and metrics["right_hand_orientation_max_deg"]
                <= acceptance["maximum_hand_orientation_max_deg"]
            ),
            "left_hand_return": (
                metrics["left_hand_return_p95_m"] <= acceptance["maximum_hand_return_p95_m"]
                and metrics["left_hand_return_max_m"] <= acceptance["maximum_hand_return_max_m"]
            ),
            "right_hand_return": (
                metrics["right_hand_return_p95_m"] <= acceptance["maximum_hand_return_p95_m"]
                and metrics["right_hand_return_max_m"] <= acceptance["maximum_hand_return_max_m"]
            ),
            "arm_joint_tracking": (
                metrics["arm_joint_target_p95_error_rad"]
                <= acceptance["maximum_arm_joint_target_p95_error_rad"]
            ),
            "base_xy_stable": (
                metrics["base_xy_excursion_max_m"] <= acceptance["maximum_base_xy_excursion_m"]
            ),
            "root_tilt": (metrics["root_tilt_max_deg"] <= acceptance["maximum_root_tilt_deg"]),
            "root_height": (
                metrics["root_height_min_m"] > acceptance["minimum_root_height_m"]
                and metrics["root_height_max_m"] < acceptance["maximum_root_height_m"]
            ),
            "arm_torque": (
                metrics["arm_torque_ratio_max"] <= acceptance["maximum_arm_torque_ratio"]
            ),
            "arm_joint_limits": (
                metrics["minimum_arm_joint_limit_margin_rad"]
                >= acceptance["minimum_arm_joint_limit_margin_rad"]
            ),
            "recurrent_state_changed": (
                metrics["hidden_state_max"] > 0.0 and metrics["cell_state_max"] > 0.0
            ),
        }

        passed = all(checks.values())

        result = {
            "passed": passed,
            "checks": checks,
            "metrics": metrics,
            "acceptance": acceptance,
            "protocol": {
                "episode_mode": "isolated",
                "warmup_steps": int(config["warmup_steps"]),
                "transient_ignore_steps": (transient_ignore),
                "no_physical_box": True,
                "no_contact_force": True,
                "virtual_box_frame": "pelvis",
                "trajectory_type": ("equal_budget_rigid_se2_small_arc"),
                "root_tilt_contract": ("not_worse_than_validated_s1_06_envelope"),
                "yaw_acceptance_contract": ("derived_from_separation_vector_error_budget"),
            },
        }

        args.output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output_json.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )
        print(
            "STAGE1_S1_07B_VIRTUAL_BOX_SMALL_ARC: "
            + ("PASS" if passed else "FAIL")
            + f" case={case_name}",
            flush=True,
        )

        if not passed:
            failed = sorted(name for name, value in checks.items() if value is not True)
            raise RuntimeError(f"Failed checks: {failed}")

    finally:
        env.close()


if __name__ == "__main__":
    pending_error: BaseException | None = None

    try:
        main()
    except BaseException as error:
        pending_error = error
    finally:
        try:
            simulation_app.close()
        except SystemExit:
            pass

    if pending_error is not None:
        raise pending_error
