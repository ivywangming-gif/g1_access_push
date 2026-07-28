"""Stateful MDP terms for S2-03T contact establishment training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import time_out as isaac_time_out

from g1_access_push.sim.stage2.s2_03t_actions import ARM_JOINT_NAMES, ArmResidualAction


CONTACT_FORCE_THRESHOLD_N = 1.0
FORCE_PEAK_THRESHOLD_N = 9.81
PALM_IMPULSE_THRESHOLD_NS = 0.981
COMBINED_IMPULSE_THRESHOLD_NS = 1.962
FORCE_RATE_THRESHOLD_NPS = 490.5
BOX_LINEAR_SPEED_THRESHOLD_MPS = 0.005
BOX_ANGULAR_SPEED_THRESHOLD_RADPS = 0.008726646259971648
BOX_TRANSLATION_THRESHOLD_M = 0.005
BOX_YAW_THRESHOLD_RAD = 0.008726646259971648
BASE_EXCURSION_THRESHOLD_M = 0.05
ROOT_HEIGHT_MIN_M = 0.5
ROOT_HEIGHT_MAX_M = 1.0
ROOT_TILT_MAX_DEG = 6.2075676918029785
ARM_MARGIN_MIN_RAD = 0.10
ARM_TORQUE_RATIO_MAX = 1.001
VERIFY_STEPS = 20
HOLD_STEPS = 100
CONTACT_LOSS_GRACE_STEPS = 5
SINGLE_HAND_MAX_STEPS = 25
IMPULSE_WINDOW_STEPS = 5
PRECONTACT_GAP_M = 0.06
PALM_SUPPORT_OFFSET_M = 0.44023889869451527
REAR_FACE_X_M = -0.6
DESIRED_PALM_QUATERNION_OBJECT_WXYZ = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)
CONTACT_TARGETS_OBJECT = (
    (REAR_FACE_X_M - PALM_SUPPORT_OFFSET_M, 0.15, 0.02),
    (REAR_FACE_X_M - PALM_SUPPORT_OFFSET_M, -0.15, 0.02),
)


def _quat_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrapped_delta(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value - reference), torch.cos(value - reference))


class S203TRuntimeState:
    """Compute physical metrics once per control step and own episode history."""

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.robot = env.scene["robot"]
        self.box = env.scene["box"]
        self.palms = env.scene["hand_frames"]
        self.left_sensor = env.scene["left_palm_box_contact"]
        self.right_sensor = env.scene["right_palm_box_contact"]
        self.robot_box_sensor = env.scene["robot_box_contact"]
        self.arm: ArmResidualAction = env.action_manager.get_term("arm_residual")
        self.arm_ids, names = self.robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        if tuple(names) != ARM_JOINT_NAMES:
            raise RuntimeError(f"runtime arm order mismatch: {names}")
        palm_ids, palm_names = self.robot.find_bodies(
            ["left_hand_palm_link", "right_hand_palm_link"], preserve_order=True
        )
        if palm_names != ["left_hand_palm_link", "right_hand_palm_link"]:
            raise RuntimeError(f"runtime palm order mismatch: {palm_names}")
        self.palm_body_ids = palm_ids
        self.arm_limits = self.robot.data.joint_pos_limits[:, self.arm_ids]
        self.arm_effort_limits = self.robot.data.joint_effort_limits[:, self.arm_ids].abs().clamp_min(1.0e-6)
        self.dt = float(env.step_dt)
        self._stamp = -1
        self.metrics: dict[str, torch.Tensor] = {}
        self.box_reference_pos = self.box.data.root_link_pos_w.clone()
        self.box_reference_yaw = _quat_yaw(self.box.data.root_link_quat_w).clone()
        self.base_reference_xy = self.robot.data.root_link_pos_w[:, :2].clone()
        self._allocate_history()

    def _allocate_history(self) -> None:
        n = self.num_envs
        self.previous_force = torch.zeros((n, 2), device=self.device)
        self.impulse = torch.zeros((n, 2), device=self.device)
        self.contact_onset_step = torch.full((n, 2), -1, dtype=torch.long, device=self.device)
        self.verify_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.hold_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.contact_verified = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.loss_streak = torch.zeros((n, 2), dtype=torch.long, device=self.device)
        self.single_hand_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.episode_min_gap = torch.full((n, 2), float("inf"), device=self.device)
        self.episode_max_force = torch.zeros((n, 2), device=self.device)
        self.episode_max_box_translation = torch.zeros(n, device=self.device)
        self.episode_max_box_yaw = torch.zeros(n, device=self.device)
        self.episode_min_root_height = torch.full((n,), float("inf"), device=self.device)
        self.episode_max_root_tilt = torch.zeros(n, device=self.device)
        self.episode_max_torque_ratio = torch.zeros(n, device=self.device)
        self.episode_min_joint_margin = torch.full((n,), float("inf"), device=self.device)
        self.episode_reward = torch.zeros(n, device=self.device)
        self.episode_contact_seen = torch.zeros(n, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.previous_force[ids] = 0.0
        self.impulse[ids] = 0.0
        self.contact_onset_step[ids] = -1
        self.verify_count[ids] = 0
        self.hold_count[ids] = 0
        self.contact_verified[ids] = False
        self.loss_streak[ids] = 0
        self.single_hand_count[ids] = 0
        self.episode_min_gap[ids] = float("inf")
        self.episode_max_force[ids] = 0.0
        self.episode_max_box_translation[ids] = 0.0
        self.episode_max_box_yaw[ids] = 0.0
        self.episode_min_root_height[ids] = float("inf")
        self.episode_max_root_tilt[ids] = 0.0
        self.episode_max_torque_ratio[ids] = 0.0
        self.episode_min_joint_margin[ids] = float("inf")
        self.episode_reward[ids] = 0.0
        self.episode_contact_seen[ids] = False
        self.box_reference_pos[ids] = self.box.data.root_link_pos_w[ids]
        self.box_reference_yaw[ids] = _quat_yaw(self.box.data.root_link_quat_w[ids])
        self.base_reference_xy[ids] = self.robot.data.root_link_pos_w[ids, :2]
        self._stamp = -1

    def ensure(self) -> dict[str, torch.Tensor]:
        stamp = int(self.env.common_step_counter)
        if self._stamp == stamp:
            return self.metrics
        self._stamp = stamp

        palm_pos_source = self.palms.data.target_pos_source
        palm_quat_source = self.palms.data.target_quat_source
        palm_pos_w, palm_quat_w = math_utils.combine_frame_transforms(
            self.robot.data.root_link_pos_w[:, None, :].expand(-1, 2, -1).reshape(-1, 3),
            self.robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
            palm_pos_source.reshape(-1, 3),
            palm_quat_source.reshape(-1, 4),
        )
        palm_pos_object, palm_quat_object = math_utils.subtract_frame_transforms(
            self.box.data.root_link_pos_w[:, None, :].expand(-1, 2, -1).reshape(-1, 3),
            self.box.data.root_link_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
            palm_pos_w,
            palm_quat_w,
        )
        palm_pos_object = palm_pos_object.reshape(self.num_envs, 2, 3)
        palm_quat_object = palm_quat_object.reshape(self.num_envs, 2, 4)
        desired_pos = torch.tensor(CONTACT_TARGETS_OBJECT, device=self.device).expand(self.num_envs, -1, -1)
        desired_quat = torch.tensor(
            DESIRED_PALM_QUATERNION_OBJECT_WXYZ, device=self.device
        ).expand(self.num_envs, 2, -1)
        position_error = desired_pos - palm_pos_object
        _, orientation_error = math_utils.compute_pose_error(
            palm_pos_object.reshape(-1, 3),
            palm_quat_object.reshape(-1, 4),
            desired_pos.reshape(-1, 3),
            desired_quat.reshape(-1, 4),
            rot_error_type="axis_angle",
        )
        orientation_error = orientation_error.reshape(self.num_envs, 2, 3)
        gaps = REAR_FACE_X_M - palm_pos_object[..., 0] - PALM_SUPPORT_OFFSET_M
        left_force = torch.linalg.vector_norm(self.left_sensor.data.force_matrix_w[:, 0, 0], dim=-1)
        right_force = torch.linalg.vector_norm(self.right_sensor.data.force_matrix_w[:, 0, 0], dim=-1)
        forces = torch.stack((left_force, right_force), dim=-1)
        contacts = forces >= CONTACT_FORCE_THRESHOLD_N
        force_rate = torch.max(torch.abs(forces - self.previous_force), dim=-1).values / self.dt

        box_translation_vector = self.box.data.root_link_pos_w - self.box_reference_pos
        box_translation = torch.linalg.vector_norm(box_translation_vector, dim=-1)
        box_yaw_delta = _wrapped_delta(
            _quat_yaw(self.box.data.root_link_quat_w), self.box_reference_yaw
        )
        box_linear_speed = torch.linalg.vector_norm(self.box.data.root_com_lin_vel_w, dim=-1)
        box_angular_speed = torch.linalg.vector_norm(self.box.data.root_com_ang_vel_w, dim=-1)
        base_excursion = torch.linalg.vector_norm(
            self.robot.data.root_link_pos_w[:, :2] - self.base_reference_xy, dim=-1
        )
        root_quat = self.robot.data.root_link_quat_w
        root_tilt = torch.rad2deg(
            torch.acos(torch.clamp(1.0 - 2.0 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2), -1.0, 1.0))
        )
        arm_q = self.robot.data.joint_pos[:, self.arm_ids]
        margin = torch.minimum(arm_q - self.arm_limits[..., 0], self.arm_limits[..., 1] - arm_q).min(dim=-1).values
        torque_ratio = (
            self.robot.data.applied_torque[:, self.arm_ids].abs() / self.arm_effort_limits
        ).max(dim=-1).values
        finite_tensors = (
            self.robot.data.root_state_w,
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.applied_torque,
            self.box.data.root_state_w,
            forces,
            self.arm.raw_actions,
        )
        finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for value in finite_tensors:
            finite &= torch.isfinite(value).reshape(self.num_envs, -1).all(dim=-1)

        forbidden = self._forbidden_overlap_mask()
        current_step = self.env.episode_length_buf.clone()
        new_onset = contacts & (self.contact_onset_step < 0)
        self.contact_onset_step[new_onset] = current_step[:, None].expand(-1, 2)[new_onset]
        within_impulse = (self.contact_onset_step >= 0) & (
            current_step[:, None] - self.contact_onset_step < IMPULSE_WINDOW_STEPS
        )
        if not getattr(self.env, "_s2_03t_bootstrap_mode", False):
            self.impulse += torch.where(within_impulse, forces * self.dt, torch.zeros_like(forces))

        physical_masks = {
            "nonfinite": ~finite,
            "forbidden_non_palm_box_collision": forbidden,
            "force_peak": forces.max(dim=-1).values > FORCE_PEAK_THRESHOLD_N,
            "palm_impulse": (self.impulse > PALM_IMPULSE_THRESHOLD_NS).any(dim=-1),
            "combined_impulse": self.impulse.sum(dim=-1) > COMBINED_IMPULSE_THRESHOLD_NS,
            "force_rate": force_rate > FORCE_RATE_THRESHOLD_NPS,
            "box_linear_speed": box_linear_speed > BOX_LINEAR_SPEED_THRESHOLD_MPS,
            "box_angular_speed": box_angular_speed > BOX_ANGULAR_SPEED_THRESHOLD_RADPS,
            "box_translation": box_translation > BOX_TRANSLATION_THRESHOLD_M,
            "box_yaw_change": box_yaw_delta.abs() > BOX_YAW_THRESHOLD_RAD,
            "base_excursion": base_excursion > BASE_EXCURSION_THRESHOLD_M,
            "root_height": (self.robot.data.root_link_pos_w[:, 2] < ROOT_HEIGHT_MIN_M)
            | (self.robot.data.root_link_pos_w[:, 2] > ROOT_HEIGHT_MAX_M),
            "root_tilt": root_tilt > ROOT_TILT_MAX_DEG,
            "arm_joint_margin": margin < ARM_MARGIN_MIN_RAD,
            "arm_torque": torque_ratio > ARM_TORQUE_RATIO_MAX,
        }
        safety_failure = torch.zeros_like(finite)
        for mask in physical_masks.values():
            safety_failure |= mask

        both = contacts.all(dim=-1)
        one = contacts.any(dim=-1) & ~both
        active = ~getattr(self.env, "_s2_03t_bootstrap_mode", False)
        if active:
            verifying = ~self.contact_verified
            self.verify_count = torch.where(verifying & both, self.verify_count + 1, torch.where(verifying, 0, self.verify_count))
            newly_verified = verifying & (self.verify_count >= VERIFY_STEPS)
            self.contact_verified |= newly_verified
            holding = self.contact_verified & ~newly_verified
            safe_hold = holding & both & ~safety_failure
            self.hold_count = torch.where(safe_hold, self.hold_count + 1, self.hold_count)
            self.loss_streak = torch.where(
                holding[:, None] & ~contacts,
                self.loss_streak + 1,
                torch.where(holding[:, None], torch.zeros_like(self.loss_streak), self.loss_streak),
            )
            self.single_hand_count = torch.where(
                verifying & one, self.single_hand_count + 1, torch.where(verifying, 0, self.single_hand_count)
            )
        contact_loss = (self.loss_streak > CONTACT_LOSS_GRACE_STEPS).any(dim=-1)
        single_hand_timeout = self.single_hand_count > SINGLE_HAND_MAX_STEPS
        success = self.hold_count >= HOLD_STEPS

        initial_gap = PRECONTACT_GAP_M
        progress = torch.clamp(1.0 - gaps.mean(dim=-1) / initial_gap, 0.0, 1.0)
        self.metrics = {
            "palm_position_error": position_error,
            "palm_orientation_error": orientation_error,
            "gaps": gaps,
            "forces": forces,
            "contacts": contacts,
            "force_rate": force_rate,
            "box_translation_vector": box_translation_vector,
            "box_translation": box_translation,
            "box_yaw_change": box_yaw_delta,
            "box_linear_velocity": self.box.data.root_com_lin_vel_w,
            "box_angular_velocity": self.box.data.root_com_ang_vel_w,
            "box_linear_speed": box_linear_speed,
            "box_angular_speed": box_angular_speed,
            "base_excursion": base_excursion,
            "root_height": self.robot.data.root_link_pos_w[:, 2],
            "root_tilt_deg": root_tilt,
            "arm_joint_margin": margin,
            "arm_torque_ratio": torque_ratio,
            "finite": finite,
            "approach_progress": progress,
            "contact_loss": contact_loss,
            "single_hand_timeout": single_hand_timeout,
            "success": success,
            "attached_hold_step": safe_hold if active else torch.zeros_like(success),
            **physical_masks,
        }
        if active:
            self.episode_min_gap = torch.minimum(self.episode_min_gap, gaps)
            self.episode_max_force = torch.maximum(self.episode_max_force, forces)
            self.episode_max_box_translation = torch.maximum(self.episode_max_box_translation, box_translation)
            self.episode_max_box_yaw = torch.maximum(self.episode_max_box_yaw, box_yaw_delta.abs())
            self.episode_min_root_height = torch.minimum(self.episode_min_root_height, self.metrics["root_height"])
            self.episode_max_root_tilt = torch.maximum(self.episode_max_root_tilt, root_tilt)
            self.episode_max_torque_ratio = torch.maximum(self.episode_max_torque_ratio, torque_ratio)
            self.episode_min_joint_margin = torch.minimum(self.episode_min_joint_margin, margin)
            self.episode_contact_seen |= both
        self.previous_force.copy_(forces)
        return self.metrics

    def _forbidden_overlap_mask(self) -> torch.Tensor:
        """Use the frozen PhysX overlap-box backend, gated by filtered contact activity."""

        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        matrix = self.robot_box_sensor.data.force_matrix_w
        if matrix is None:
            return mask
        active = torch.linalg.vector_norm(matrix[..., 0, :], dim=-1).max(dim=-1).values > 1.0e-6
        active_ids = active.nonzero(as_tuple=False).squeeze(-1).tolist()
        if not active_ids:
            return mask
        try:
            import carb
            from omni.physx import get_physx_scene_query_interface
        except ImportError as exc:  # pragma: no cover - only available under Isaac Sim
            raise RuntimeError("PHYSX_SCENE_QUERY_UNAVAILABLE") from exc
        query = get_physx_scene_query_interface()
        for env_id in active_ids:
            robot_prefix = f"/World/envs/env_{env_id}/Robot"
            allowed = {
                f"{robot_prefix}/left_hand/left_hand_palm_link",
                f"{robot_prefix}/right_hand/right_hand_palm_link",
            }
            forbidden = False

            def callback(hit: Any) -> bool:
                nonlocal forbidden
                collision = str(getattr(hit, "collision", ""))
                rigid_body = str(getattr(hit, "rigid_body", ""))
                path = rigid_body or collision
                if path.startswith(robot_prefix) and path not in allowed:
                    forbidden = True
                    return False
                return True

            position = self.box.data.root_link_pos_w[env_id]
            quaternion = self.box.data.root_link_quat_w[env_id]
            query.overlap_box(
                carb.Float3(0.6, 0.3, 0.6),
                carb.Float3(*[float(item) for item in position]),
                carb.Float4(
                    float(quaternion[1]), float(quaternion[2]), float(quaternion[3]), float(quaternion[0])
                ),
                callback,
                False,
            )
            mask[env_id] = forbidden
        return mask

    def snapshot(self, env_ids: Sequence[int] | torch.Tensor) -> dict[int, dict[str, Any]]:
        metrics = self.ensure()
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        snapshots: dict[int, dict[str, Any]] = {}
        for env_id in ids.tolist():
            reasons = [
                name
                for name, value in metrics.items()
                if name in TERMINATION_METRIC_NAMES and bool(value[env_id])
            ]
            snapshots[env_id] = {
                "episode_steps": int(self.env.episode_length_buf[env_id]),
                "success": bool(metrics["success"][env_id]),
                "time_out": bool(self.env.reset_time_outs[env_id]) if hasattr(self.env, "reset_time_outs") else False,
                "termination_reasons": reasons,
                "verify_steps": int(self.verify_count[env_id]),
                "attached_hold_steps": int(self.hold_count[env_id]),
                "minimum_surface_gap_m": [float(value) for value in self.episode_min_gap[env_id]],
                "maximum_force_n": [float(value) for value in self.episode_max_force[env_id]],
                "impulse_ns": [float(value) for value in self.impulse[env_id]],
                "maximum_box_translation_m": float(self.episode_max_box_translation[env_id]),
                "maximum_box_yaw_change_rad": float(self.episode_max_box_yaw[env_id]),
                "minimum_root_height_m": float(self.episode_min_root_height[env_id]),
                "maximum_root_tilt_deg": float(self.episode_max_root_tilt[env_id]),
                "minimum_arm_joint_margin_rad": float(self.episode_min_joint_margin[env_id]),
                "maximum_arm_torque_ratio": float(self.episode_max_torque_ratio[env_id]),
                "bilateral_contact_seen": bool(self.episode_contact_seen[env_id]),
                "finite": bool(metrics["finite"][env_id]),
            }
        return snapshots


TERMINATION_METRIC_NAMES = {
    "nonfinite",
    "forbidden_non_palm_box_collision",
    "contact_loss",
    "single_hand_timeout",
    "force_peak",
    "palm_impulse",
    "combined_impulse",
    "force_rate",
    "box_linear_speed",
    "box_angular_speed",
    "box_translation",
    "box_yaw_change",
    "base_excursion",
    "root_height",
    "root_tilt",
    "arm_joint_margin",
    "arm_torque",
}


def runtime_state(env: ManagerBasedRLEnv) -> S203TRuntimeState:
    state = getattr(env, "_s2_03t_runtime_state", None)
    if state is None:
        state = S203TRuntimeState(env)
        env._s2_03t_runtime_state = state
    return state


def policy_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = runtime_state(env)
    metric = state.ensure()
    robot = state.robot
    arm = state.arm

    def symmetric_clip(value: torch.Tensor, bound: float) -> torch.Tensor:
        return torch.clamp(value, -bound, bound)

    components = (
        symmetric_clip(robot.data.root_ang_vel_b, 5.0),
        symmetric_clip(robot.data.projected_gravity_b, 1.0),
        symmetric_clip(robot.data.joint_pos[:, state.arm_ids] - arm.reference, 1.0),
        symmetric_clip(robot.data.joint_vel[:, state.arm_ids], 10.0),
        symmetric_clip(metric["palm_position_error"][:, 0], 0.10),
        symmetric_clip(metric["palm_position_error"][:, 1], 0.10),
        symmetric_clip(metric["palm_orientation_error"][:, 0], 0.5),
        symmetric_clip(metric["palm_orientation_error"][:, 1], 0.5),
        torch.clamp(metric["gaps"], -0.02, 0.10),
        torch.clamp(metric["forces"], 0.0, 9.81),
        metric["contacts"].to(dtype=robot.data.joint_pos.dtype),
        symmetric_clip(metric["box_translation_vector"], 0.005),
        symmetric_clip(metric["box_yaw_change"].unsqueeze(-1), BOX_YAW_THRESHOLD_RAD),
        symmetric_clip(metric["box_linear_velocity"], 0.005),
        symmetric_clip(metric["box_angular_velocity"], BOX_ANGULAR_SPEED_THRESHOLD_RADPS),
        torch.clamp(metric["approach_progress"].unsqueeze(-1), 0.0, 1.0),
        symmetric_clip(arm.raw_actions, 1.0),
    )
    observation = torch.cat(components, dim=-1)
    if tuple(observation.shape) != (env.num_envs, 77):
        raise RuntimeError(f"S2-03T observation shape mismatch: {tuple(observation.shape)}")
    if not bool(torch.isfinite(observation).all()):
        raise RuntimeError("S2-03T observation contains NaN/Inf")
    return observation


# Reward formulas.  The config owns the frozen weights.
def bilateral_contact_verify_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["contacts"].all(dim=-1).float()


def attached_hold_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["attached_hold_step"].float()


def success_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["success"].float()


def symmetric_gap_closure(env: ManagerBasedRLEnv) -> torch.Tensor:
    gaps = runtime_state(env).ensure()["gaps"]
    return torch.exp(-50.0 * torch.abs(gaps).sum(dim=-1))


def hand_position_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.linalg.vector_norm(runtime_state(env).ensure()["palm_position_error"], dim=-1).sum(dim=-1)


def hand_orientation_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.linalg.vector_norm(runtime_state(env).ensure()["palm_orientation_error"], dim=-1).sum(dim=-1)


def force_imbalance(env: ManagerBasedRLEnv) -> torch.Tensor:
    forces = runtime_state(env).ensure()["forces"]
    return torch.abs(forces[:, 0] - forces[:, 1])


def box_translation(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["box_translation"]


def box_yaw_change(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["box_yaw_change"].abs()


def force_impulse_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = runtime_state(env)
    metric = state.ensure()
    return (
        metric["forces"].max(dim=-1).values / FORCE_PEAK_THRESHOLD_N
        + state.impulse.sum(dim=-1) / COMBINED_IMPULSE_THRESHOLD_NS
        + metric["force_rate"] / FORCE_RATE_THRESHOLD_NPS
    )


def root_tilt(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["root_tilt_deg"]


def joint_limit_margin(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(ARM_MARGIN_MIN_RAD - runtime_state(env).ensure()["arm_joint_margin"], min=0.0)


def torque_ratio(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(runtime_state(env).ensure()["arm_torque_ratio"] - 1.0, min=0.0)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(runtime_state(env).arm.raw_actions).sum(dim=-1)


def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(runtime_state(env).arm.action_delta).sum(dim=-1)


def forbidden_collision_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["forbidden_non_palm_box_collision"].float()


def contact_timeout_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    return isaac_time_out(env).float()


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    if getattr(env, "_s2_03t_bootstrap_mode", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return runtime_state(env).ensure()["success"]


def metric_termination(env: ManagerBasedRLEnv, metric_name: str) -> torch.Tensor:
    if getattr(env, "_s2_03t_bootstrap_mode", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return runtime_state(env).ensure()[metric_name]


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    if getattr(env, "_s2_03t_bootstrap_mode", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return isaac_time_out(env)
