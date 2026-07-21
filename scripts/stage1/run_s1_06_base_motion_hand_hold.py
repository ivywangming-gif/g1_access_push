"""Validate base motion while both palms remain fixed relative to the pelvis."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=Path,
    default=PROJECT_ROOT / "configs/stage1/s1_06_base_motion_hand_hold.yaml",
)
parser.add_argument("--output_json", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402

from g1_access_push.sim.stage1.no_box_env_cfg import (  # noqa: E402
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)
from g1_access_push.sim.stage1.recurrent_no_box_env_cfg import (  # noqa: E402
    G1Stage1NoBoxRecurrentEnvCfg,
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
    dots = torch.sum(quaternions * reference, dim=-1).abs()
    dots = torch.clamp(dots, 0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dots))


def root_tilt_deg(quaternions: torch.Tensor) -> torch.Tensor:
    """Return roll/pitch tilt while ignoring yaw."""

    x = quaternions[..., 1]
    y = quaternions[..., 2]

    cosine = 1.0 - 2.0 * (x.square() + y.square())
    cosine = torch.clamp(cosine, -1.0, 1.0)

    return torch.rad2deg(torch.acos(cosine))


def yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Return yaw from a wxyz quaternion."""

    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]

    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


def wrapped_angle_difference(
    final_angle: torch.Tensor,
    initial_angle: torch.Tensor,
) -> torch.Tensor:
    difference = final_angle - initial_angle

    return torch.atan2(
        torch.sin(difference),
        torch.cos(difference),
    )


def measured_axis(
    robot,
    axis: str,
) -> torch.Tensor:
    if axis == "x":
        return robot.data.root_lin_vel_b[0, 0].clone()

    if axis == "y":
        return robot.data.root_lin_vel_b[0, 1].clone()

    if axis == "wz":
        return robot.data.root_ang_vel_b[0, 2].clone()

    raise ValueError(f"Unsupported evaluation axis: {axis}")


def main() -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    acceptance = config["acceptance"]

    cfg = G1Stage1NoBoxRecurrentEnvCfg()
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

        lower_term = env.action_manager.get_term(
            "lower_body_joint_pos"
        )

        upper_dim = upper_slice.stop - upper_slice.start

        arm_joint_ids, resolved_arm_names = robot.find_joints(
            LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
        )

        frame_names = hand_frames.data.target_frame_names
        left_frame_id = frame_names.index("left_hand_palm")
        right_frame_id = frame_names.index("right_hand_palm")

        actions = torch.zeros(
            (1, env.action_manager.total_action_dim),
            dtype=torch.float32,
            device=env.device,
        )
        zero_upper = torch.zeros(
            upper_dim,
            dtype=torch.float32,
            device=env.device,
        )

        def step_env(command: torch.Tensor) -> None:
            actions.zero_()
            actions[:, upper_slice] = zero_upper
            actions[:, lower_slice] = command
            env.step(actions)

        zero_lower = torch.tensor(
            [0.0, 0.0, 0.0, 0.72],
            dtype=torch.float32,
            device=env.device,
        )

        for step in range(int(config["warmup_steps"])):
            step_env(zero_lower)

            tensors = (
                robot.data.root_state_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
                lower_term.policy_actions,
                lower_term.last_policy_input,
            )

            if not all(
                bool(torch.isfinite(value).all())
                for value in tensors
            ):
                raise RuntimeError(
                    f"Non-finite warmup state at step {step}"
                )

        baseline_palms_pos = (
            hand_frames.data.target_pos_source[0].clone()
        )
        baseline_palms_quat = (
            hand_frames.data.target_quat_source[0].clone()
        )

        left_position_errors: list[torch.Tensor] = []
        right_position_errors: list[torch.Tensor] = []
        left_orientation_errors: list[torch.Tensor] = []
        right_orientation_errors: list[torch.Tensor] = []
        root_heights: list[torch.Tensor] = []
        root_tilts: list[torch.Tensor] = []
        torque_ratios: list[torch.Tensor] = []

        effort_limits = (
            robot.data.joint_effort_limits[0, arm_joint_ids]
            .abs()
            .clamp_min(1.0e-6)
        )

        transient_ignore = int(
            config["transient_ignore_steps"]
        )

        segment_metrics: dict[str, dict] = {}
        segment_checks: dict[str, bool] = {}

        all_numeric_finite = True

        for segment in config["segments"]:
            name = str(segment["name"])
            steps = int(segment["steps"])

            command = torch.tensor(
                segment["command"],
                dtype=torch.float32,
                device=env.device,
            )

            evaluate_axes = {
                str(axis): float(target)
                for axis, target in segment.get(
                    "evaluate_axes",
                    {},
                ).items()
            }

            if any(
                axis not in {"x", "y", "wz"}
                for axis in evaluate_axes
            ):
                raise ValueError(
                    f"Invalid evaluate_axes in segment {name}: "
                    f"{sorted(evaluate_axes)}"
                )

            if evaluate_axes and steps <= transient_ignore:
                raise ValueError(
                    f"Segment {name} must be longer than "
                    "transient_ignore_steps."
                )

            measured_values: dict[
                str,
                list[torch.Tensor],
            ] = {
                axis: []
                for axis in evaluate_axes
            }

            initial_yaw = yaw_from_quaternion(
                robot.data.root_quat_w[0]
            ).clone()

            for step in range(steps):
                step_env(command)

                palms_pos = (
                    hand_frames.data.target_pos_source[0]
                )
                palms_quat = (
                    hand_frames.data.target_quat_source[0]
                )
                root_quat = robot.data.root_quat_w[0]

                tensors = (
                    robot.data.root_state_w,
                    robot.data.joint_pos,
                    robot.data.joint_vel,
                    robot.data.applied_torque,
                    palms_pos,
                    palms_quat,
                    lower_term.policy_actions,
                    lower_term.processed_actions,
                    lower_term.last_policy_input,
                )

                if not all(
                    bool(torch.isfinite(value).all())
                    for value in tensors
                ):
                    all_numeric_finite = False
                    raise RuntimeError(
                        f"Non-finite state in {name}, step {step}"
                    )

                left_position_errors.append(
                    torch.linalg.vector_norm(
                        palms_pos[left_frame_id]
                        - baseline_palms_pos[left_frame_id]
                    )
                )
                right_position_errors.append(
                    torch.linalg.vector_norm(
                        palms_pos[right_frame_id]
                        - baseline_palms_pos[right_frame_id]
                    )
                )

                left_orientation_errors.append(
                    quaternion_error_deg(
                        palms_quat[
                            left_frame_id
                        ].unsqueeze(0),
                        baseline_palms_quat[left_frame_id],
                    )[0]
                )
                right_orientation_errors.append(
                    quaternion_error_deg(
                        palms_quat[
                            right_frame_id
                        ].unsqueeze(0),
                        baseline_palms_quat[right_frame_id],
                    )[0]
                )

                root_heights.append(
                    robot.data.root_pos_w[0, 2].clone()
                )
                root_tilts.append(
                    root_tilt_deg(
                        root_quat.unsqueeze(0)
                    )[0]
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

                if step >= transient_ignore:
                    for axis in evaluate_axes:
                        measured_values[axis].append(
                            measured_axis(robot, axis)
                        )

            final_yaw = yaw_from_quaternion(
                robot.data.root_quat_w[0]
            ).clone()

            axes_metrics: dict[str, dict] = {}

            for axis, target in evaluate_axes.items():
                measured = torch.stack(
                    measured_values[axis]
                )
                median_measured = float(
                    torch.median(measured).item()
                )

                errors = (measured - target).abs()
                error_p95 = p95(errors)

                response_ratio = (
                    abs(median_measured)
                    / max(abs(target), 1.0e-6)
                )
                sign_correct = (
                    median_measured * target > 0.0
                )

                if axis in {"x", "y"}:
                    maximum_error = acceptance[
                        "maximum_linear_velocity_p95_error_m_s"
                    ]
                else:
                    maximum_error = acceptance[
                        "maximum_yaw_rate_p95_error_rad_s"
                    ]

                axis_passed = (
                    sign_correct
                    and response_ratio
                    >= acceptance[
                        "minimum_velocity_response_ratio"
                    ]
                    and error_p95 <= maximum_error
                )

                axes_metrics[axis] = {
                    "target": target,
                    "median_measured": median_measured,
                    "response_ratio": response_ratio,
                    "absolute_error_p95": error_p95,
                    "sign_correct": sign_correct,
                    "passed": axis_passed,
                }

                segment_checks[
                    f"velocity_{name}_{axis}"
                ] = axis_passed

            yaw_delta = float(
                wrapped_angle_difference(
                    final_yaw,
                    initial_yaw,
                ).item()
            )

            yaw_displacement_required = bool(
                segment.get(
                    "require_yaw_displacement",
                    False,
                )
            )

            yaw_displacement_passed = True

            if yaw_displacement_required:
                target_wz = evaluate_axes.get("wz")

                if target_wz is None:
                    raise ValueError(
                        f"Segment {name} requires yaw "
                        "displacement but has no wz target."
                    )

                yaw_displacement_passed = (
                    yaw_delta * target_wz > 0.0
                    and abs(yaw_delta)
                    >= acceptance[
                        "minimum_arc_yaw_displacement_rad"
                    ]
                )

                segment_checks[
                    f"yaw_displacement_{name}"
                ] = yaw_displacement_passed

            segment_metrics[name] = {
                "steps": steps,
                "duration_s": steps * env.step_dt,
                "command": list(
                    map(float, segment["command"])
                ),
                "axes": axes_metrics,
                "yaw_delta_rad": yaw_delta,
                "yaw_displacement_required": (
                    yaw_displacement_required
                ),
                "yaw_displacement_passed": (
                    yaw_displacement_passed
                ),
            }

        left_position_errors_t = torch.stack(
            left_position_errors
        )
        right_position_errors_t = torch.stack(
            right_position_errors
        )
        left_orientation_errors_t = torch.stack(
            left_orientation_errors
        )
        right_orientation_errors_t = torch.stack(
            right_orientation_errors
        )
        root_heights_t = torch.stack(root_heights)
        root_tilts_t = torch.stack(root_tilts)
        torque_ratios_t = torch.stack(torque_ratios)

        metrics = {
            "test_id": config["test_id"],
            "seed": cfg.seed,
            "step_dt_s": env.step_dt,
            "total_motion_steps": len(root_heights),
            "total_motion_duration_s": (
                len(root_heights) * env.step_dt
            ),
            "resolved_arm_joint_names": (
                resolved_arm_names
            ),
            "segments": segment_metrics,
            "left_hand_position_p95_m": p95(
                left_position_errors_t
            ),
            "left_hand_position_max_m": float(
                left_position_errors_t.max().item()
            ),
            "right_hand_position_p95_m": p95(
                right_position_errors_t
            ),
            "right_hand_position_max_m": float(
                right_position_errors_t.max().item()
            ),
            "left_hand_orientation_p95_deg": p95(
                left_orientation_errors_t
            ),
            "left_hand_orientation_max_deg": float(
                left_orientation_errors_t.max().item()
            ),
            "right_hand_orientation_p95_deg": p95(
                right_orientation_errors_t
            ),
            "right_hand_orientation_max_deg": float(
                right_orientation_errors_t.max().item()
            ),
            "root_height_min_m": float(
                root_heights_t.min().item()
            ),
            "root_height_max_m": float(
                root_heights_t.max().item()
            ),
            "root_tilt_max_deg": float(
                root_tilts_t.max().item()
            ),
            "arm_torque_ratio_max": float(
                torque_ratios_t.max().item()
            ),
            "hidden_state_max": (
                lower_term.recurrent_state_max
            ),
            "cell_state_max": (
                lower_term.recurrent_cell_max
            ),
            "all_numeric_finite": all_numeric_finite,
        }

        checks = {
            **segment_checks,
            "finite": all_numeric_finite,
            "left_hand_position": (
                metrics["left_hand_position_p95_m"]
                <= acceptance[
                    "maximum_hand_position_p95_m"
                ]
                and metrics["left_hand_position_max_m"]
                <= acceptance[
                    "maximum_hand_position_max_m"
                ]
            ),
            "right_hand_position": (
                metrics["right_hand_position_p95_m"]
                <= acceptance[
                    "maximum_hand_position_p95_m"
                ]
                and metrics["right_hand_position_max_m"]
                <= acceptance[
                    "maximum_hand_position_max_m"
                ]
            ),
            "left_hand_orientation": (
                metrics[
                    "left_hand_orientation_p95_deg"
                ]
                <= acceptance[
                    "maximum_hand_orientation_p95_deg"
                ]
                and metrics[
                    "left_hand_orientation_max_deg"
                ]
                <= acceptance[
                    "maximum_hand_orientation_max_deg"
                ]
            ),
            "right_hand_orientation": (
                metrics[
                    "right_hand_orientation_p95_deg"
                ]
                <= acceptance[
                    "maximum_hand_orientation_p95_deg"
                ]
                and metrics[
                    "right_hand_orientation_max_deg"
                ]
                <= acceptance[
                    "maximum_hand_orientation_max_deg"
                ]
            ),
            "root_height": (
                metrics["root_height_min_m"]
                > acceptance["minimum_root_height_m"]
                and metrics["root_height_max_m"]
                < acceptance["maximum_root_height_m"]
            ),
            "root_tilt": (
                metrics["root_tilt_max_deg"]
                <= acceptance["maximum_root_tilt_deg"]
            ),
            "arm_torque": (
                metrics["arm_torque_ratio_max"]
                <= acceptance[
                    "maximum_arm_torque_ratio"
                ]
            ),
            "recurrent_state_changed": (
                metrics["hidden_state_max"] > 0.0
                and metrics["cell_state_max"] > 0.0
            ),
        }

        passed = all(checks.values())

        result = {
            "passed": passed,
            "checks": checks,
            "metrics": metrics,
            "acceptance": acceptance,
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
            "STAGE1_S1_06_RECURRENT_BASE_MOTION_HAND_HOLD: "
            + ("PASS" if passed else "FAIL"),
            flush=True,
        )

        if not passed:
            failed = [
                name
                for name, value in checks.items()
                if not value
            ]
            raise RuntimeError(
                f"Failed checks: {failed}"
            )

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
