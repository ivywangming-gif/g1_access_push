"""Validate straight virtual-box surface tracking with both palms."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=Path,
    default=(PROJECT_ROOT / "configs/stage1/s1_07a_virtual_box_straight.yaml"),
)
parser.add_argument("--output_json", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402

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
    straight_out_hold_back_displacement,
    virtual_surface_contact_targets,
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


def main() -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if config.get("episode_mode") != "isolated":
        raise ValueError("S1-07A requires episode_mode='isolated'.")

    trajectory = config["trajectory"]
    acceptance = config["acceptance"]

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
            raise RuntimeError(f"Unexpected S1-07A action contract: {resolved_action_dims}")

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

        if {
            left_frame_id,
            right_frame_id,
        } != {0, 1}:
            raise RuntimeError(f"Unexpected hand-frame ordering: {frame_names}")

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

        def step_zero_targets() -> None:
            actions.zero_()
            actions[:, waist_slice] = 0.0
            actions[:, lower_slice] = lower_command
            env.step(actions)

        for step in range(int(config["warmup_steps"])):
            step_zero_targets()

            warmup_tensors = (
                robot.data.root_state_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
                lower_term.policy_actions,
                lower_term.last_policy_input,
                hand_frames.data.target_pos_source,
                hand_frames.data.target_quat_source,
            )

            if not all(bool(torch.isfinite(value).all()) for value in warmup_tensors):
                raise RuntimeError(f"Non-finite warmup state at step {step}.")

        baseline_root_pos = robot.data.root_pos_w[0].clone()
        baseline_root_quat = robot.data.root_quat_w[0].clone()

        baseline_palms_pos = hand_frames.data.target_pos_source[0].clone()
        baseline_palms_quat = hand_frames.data.target_quat_source[0].clone()

        baseline_midpoint = baseline_palms_pos.mean(dim=0)
        baseline_separation = baseline_palms_pos[right_frame_id] - baseline_palms_pos[left_frame_id]

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
            raise ValueError("hold_steps must exceed transient_ignore_steps.")

        if settle_steps <= transient_ignore:
            raise ValueError("settle_steps must exceed transient_ignore_steps.")

        profile_steps = move_out_steps + hold_steps + move_back_steps
        total_steps = profile_steps + settle_steps

        position_errors: list[torch.Tensor] = []
        orientation_errors: list[torch.Tensor] = []
        palm_positions: list[torch.Tensor] = []
        separation_errors: list[torch.Tensor] = []
        root_positions: list[torch.Tensor] = []
        root_quaternions: list[torch.Tensor] = []
        root_tilts: list[torch.Tensor] = []
        arm_target_errors: list[torch.Tensor] = []
        arm_velocities: list[torch.Tensor] = []
        torque_ratios: list[torch.Tensor] = []
        joint_limit_margins: list[torch.Tensor] = []
        position_corrections: list[torch.Tensor] = []
        orientation_corrections: list[torch.Tensor] = []

        maximum_position_correction = float(trajectory["maximum_position_correction_m"])
        maximum_orientation_correction = float(trajectory["maximum_orientation_correction_rad"])

        all_numeric_finite = True

        for step in range(total_steps):
            if step < profile_steps:
                displacement = straight_out_hold_back_displacement(
                    step,
                    amplitude_m=float(trajectory["amplitude_m"]),
                    move_out_steps=move_out_steps,
                    hold_steps=hold_steps,
                    move_back_steps=move_back_steps,
                )
            else:
                displacement = 0.0

            translation = torch.tensor(
                [displacement, 0.0, 0.0],
                dtype=torch.float32,
                device=env.device,
            )

            desired_positions = virtual_surface_contact_targets(
                baseline_palms_pos,
                translation,
            )
            desired_quaternions = baseline_palms_quat

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

            actual_positions = hand_frames.data.target_pos_source[0]
            actual_quaternions = hand_frames.data.target_quat_source[0]

            current_position_errors = torch.linalg.vector_norm(
                actual_positions - desired_positions,
                dim=-1,
            )
            current_orientation_errors = quaternion_error_deg(
                actual_quaternions,
                desired_quaternions,
            )

            actual_separation = actual_positions[right_frame_id] - actual_positions[left_frame_id]
            separation_error = torch.linalg.vector_norm(actual_separation - baseline_separation)

            current_arm_positions = robot.data.joint_pos[
                0,
                arm_joint_ids,
            ]
            current_arm_targets = robot.data.joint_pos_target[
                0,
                arm_joint_ids,
            ]
            current_arm_velocities = robot.data.joint_vel[
                0,
                arm_joint_ids,
            ]
            current_torque_ratio = (
                robot.data.applied_torque[
                    0,
                    arm_joint_ids,
                ].abs()
                / effort_limits
            )

            lower_margin = current_arm_positions - joint_limits[:, 0]
            upper_margin = joint_limits[:, 1] - current_arm_positions
            current_joint_margin = torch.minimum(
                lower_margin,
                upper_margin,
            )

            tensors = (
                robot.data.root_state_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
                robot.data.applied_torque,
                actual_positions,
                actual_quaternions,
                desired_positions,
                desired_quaternions,
                position_correction,
                orientation_correction,
                lower_term.policy_actions,
                lower_term.processed_actions,
                lower_term.last_policy_input,
            )

            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                all_numeric_finite = False
                raise RuntimeError(f"Non-finite S1-07A state at step {step}.")

            position_errors.append(current_position_errors.clone())
            orientation_errors.append(current_orientation_errors.clone())
            palm_positions.append(actual_positions.clone())
            separation_errors.append(separation_error.clone())
            root_positions.append(robot.data.root_pos_w[0].clone())
            root_quaternions.append(robot.data.root_quat_w[0].clone())
            root_tilts.append(root_tilt_deg(robot.data.root_quat_w[0].unsqueeze(0))[0])
            arm_target_errors.append((current_arm_positions - current_arm_targets).abs().clone())
            arm_velocities.append(current_arm_velocities.clone())
            torque_ratios.append(current_torque_ratio.clone())
            joint_limit_margins.append(current_joint_margin.clone())
            position_corrections.append(position_correction.clone())
            orientation_corrections.append(orientation_correction.clone())

        position_errors_t = torch.stack(position_errors)
        orientation_errors_t = torch.stack(orientation_errors)
        palm_positions_t = torch.stack(palm_positions)
        separation_errors_t = torch.stack(separation_errors)
        root_positions_t = torch.stack(root_positions)
        root_quaternions_t = torch.stack(root_quaternions)
        root_tilts_t = torch.stack(root_tilts)
        arm_target_errors_t = torch.stack(arm_target_errors)
        arm_velocities_t = torch.stack(arm_velocities)
        torque_ratios_t = torch.stack(torque_ratios)
        joint_limit_margins_t = torch.stack(joint_limit_margins)
        position_corrections_t = torch.stack(position_corrections)
        orientation_corrections_t = torch.stack(orientation_corrections)

        arm_accelerations_t = (
            torch.diff(
                arm_velocities_t,
                dim=0,
            )
            / env.step_dt
        )
        arm_jerks_t = (
            torch.diff(
                arm_accelerations_t,
                dim=0,
            )
            / env.step_dt
        )

        midpoints_t = palm_positions_t.mean(dim=1)
        midpoint_offsets_t = midpoints_t - baseline_midpoint.unsqueeze(0)

        hold_start = move_out_steps + transient_ignore
        hold_end = move_out_steps + hold_steps

        settle_start = profile_steps + transient_ignore
        settle_end = total_steps

        steady_hold_x = midpoint_offsets_t[
            hold_start:hold_end,
            0,
        ]
        steady_hold_x_error = (steady_hold_x - float(trajectory["amplitude_m"])).abs()

        settle_left_return = torch.linalg.vector_norm(
            palm_positions_t[
                settle_start:settle_end,
                left_frame_id,
            ]
            - baseline_palms_pos[left_frame_id],
            dim=-1,
        )
        settle_right_return = torch.linalg.vector_norm(
            palm_positions_t[
                settle_start:settle_end,
                right_frame_id,
            ]
            - baseline_palms_pos[right_frame_id],
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

        left_position_errors = position_errors_t[
            :,
            left_frame_id,
        ]
        right_position_errors = position_errors_t[
            :,
            right_frame_id,
        ]
        left_orientation_errors = orientation_errors_t[
            :,
            left_frame_id,
        ]
        right_orientation_errors = orientation_errors_t[
            :,
            right_frame_id,
        ]

        metrics = {
            "test_id": config["test_id"],
            "seed": cfg.seed,
            "step_dt_s": env.step_dt,
            "total_command_steps": total_steps,
            "total_command_duration_s": (total_steps * env.step_dt),
            "action_terms": action_names,
            "action_dims": action_dims,
            "resolved_action_dims": (resolved_action_dims),
            "resolved_arm_joint_names": (resolved_arm_names),
            "hand_frame_names": frame_names,
            "baseline_hand_positions_in_pelvis": (baseline_palms_pos.detach().cpu().tolist()),
            "baseline_hand_quaternions_in_pelvis": (baseline_palms_quat.detach().cpu().tolist()),
            "baseline_contact_separation_m": float(
                torch.linalg.vector_norm(baseline_separation).item()
            ),
            "commanded_amplitude_m": float(trajectory["amplitude_m"]),
            "steady_hold_midpoint_x_median_m": float(torch.median(steady_hold_x).item()),
            "steady_hold_midpoint_x_p95_error_m": p95(steady_hold_x_error),
            "left_hand_position_p95_m": p95(left_position_errors),
            "left_hand_position_max_m": float(left_position_errors.max().item()),
            "right_hand_position_p95_m": p95(right_position_errors),
            "right_hand_position_max_m": float(right_position_errors.max().item()),
            "left_hand_orientation_p95_deg": p95(left_orientation_errors),
            "left_hand_orientation_max_deg": float(left_orientation_errors.max().item()),
            "right_hand_orientation_p95_deg": p95(right_orientation_errors),
            "right_hand_orientation_max_deg": float(right_orientation_errors.max().item()),
            "contact_separation_p95_error_m": p95(separation_errors_t),
            "contact_separation_max_error_m": float(separation_errors_t.max().item()),
            "left_hand_return_p95_m": p95(settle_left_return),
            "left_hand_return_max_m": float(settle_left_return.max().item()),
            "right_hand_return_p95_m": p95(settle_right_return),
            "right_hand_return_max_m": float(settle_right_return.max().item()),
            "arm_joint_target_p95_error_rad": p95(arm_target_errors_t),
            "arm_joint_target_max_error_rad": float(arm_target_errors_t.max().item()),
            "arm_joint_velocity_p95_rad_s": p95(arm_velocities_t.abs()),
            "arm_joint_velocity_max_rad_s": float(arm_velocities_t.abs().max().item()),
            "arm_joint_acceleration_p95_rad_s2": p95(arm_accelerations_t.abs()),
            "arm_joint_acceleration_max_rad_s2": float(arm_accelerations_t.abs().max().item()),
            "arm_joint_jerk_p95_rad_s3": p95(arm_jerks_t.abs()),
            "arm_joint_jerk_max_rad_s3": float(arm_jerks_t.abs().max().item()),
            "arm_torque_ratio_max": float(torque_ratios_t.max().item()),
            "minimum_arm_joint_limit_margin_rad": float(joint_limit_margins_t.min().item()),
            "base_xy_excursion_max_m": float(root_xy_excursions.max().item()),
            "root_orientation_error_max_deg": float(root_orientation_errors.max().item()),
            "root_tilt_max_deg": float(root_tilts_t.max().item()),
            "root_height_min_m": float(root_positions_t[:, 2].min().item()),
            "root_height_max_m": float(root_positions_t[:, 2].max().item()),
            "position_correction_max_m": float(
                torch.linalg.vector_norm(
                    position_corrections_t,
                    dim=-1,
                )
                .max()
                .item()
            ),
            "orientation_correction_max_rad": float(
                torch.linalg.vector_norm(
                    orientation_corrections_t,
                    dim=-1,
                )
                .max()
                .item()
            ),
            "hidden_state_max": (lower_term.recurrent_state_max),
            "cell_state_max": (lower_term.recurrent_cell_max),
            "all_numeric_finite": (all_numeric_finite),
        }

        checks = {
            "action_contract": (resolved_action_dims == expected_action_dims),
            "finite": all_numeric_finite,
            "virtual_box_translation": (
                metrics["steady_hold_midpoint_x_median_m"]
                >= acceptance["minimum_steady_hold_midpoint_x_m"]
                and metrics["steady_hold_midpoint_x_p95_error_m"]
                <= acceptance["maximum_steady_hold_midpoint_x_p95_error_m"]
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
            "contact_separation": (
                metrics["contact_separation_p95_error_m"]
                <= acceptance["maximum_contact_separation_p95_error_m"]
                and metrics["contact_separation_max_error_m"]
                <= acceptance["maximum_contact_separation_max_error_m"]
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
                "root_tilt_contract": ("not_worse_than_validated_s1_06_envelope"),
                "root_full_orientation_diagnostic_only": True,
                "trajectory_type": ("cosine_straight_out_hold_back"),
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
            "STAGE1_S1_07A_VIRTUAL_BOX_STRAIGHT: " + ("PASS" if passed else "FAIL"),
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
            # Isaac Sim shutdown must not mask a failure.
            pass

    if pending_error is not None:
        raise pending_error
