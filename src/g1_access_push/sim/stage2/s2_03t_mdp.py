"""Stateful MDP terms for S2-03T contact establishment training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import time_out as isaac_time_out

from g1_access_push.sim.stage2.s2_03t_actions import ARM_JOINT_NAMES, ArmResidualAction
from g1_access_push.stage2.s2_03t_tensor_contract import filtered_contact_activity

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
SOFT_FORCE_MAXIMUM_N = 3.0
GAP_PROGRESS_DEADBAND_M = 1.0e-5
GAP_PROGRESS_CLIP_M = 1.0e-3
ROOT_RISK_DEADBAND_DEG = 5.0
MODE_APPROACH = 0
MODE_CONTACT_ACQUIRE = 1
MODE_VERIFY_HOLD = 2
MODE_COUNT = 3
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
        self.arm_effort_limits = (
            self.robot.data.joint_effort_limits[:, self.arm_ids].abs().clamp_min(1.0e-6)
        )
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
        self.last_history_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.gap_history_initialized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.previous_gap_sum = torch.zeros(n, device=self.device)
        self.reward_mode = torch.full((n,), MODE_APPROACH, dtype=torch.long, device=self.device)
        self.contact_acquire_started = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.bilateral_onset_seen = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.left_onset_event = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.right_onset_event = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.bilateral_onset_event = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.gap_progress_value = torch.zeros(n, device=self.device)
        self.episode_contact_retention_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._allocate_training_interval()

    def _allocate_training_interval(self) -> None:
        def zero_scalar() -> torch.Tensor:
            return torch.zeros((), dtype=torch.float64, device=self.device)

        self._training_interval = {
            "sample_count": zero_scalar(),
            "bilateral_contact_count": zero_scalar(),
            "bilateral_onset_count": zero_scalar(),
            "verify_count": zero_scalar(),
            "hold_count": zero_scalar(),
            "contact_retention_count": zero_scalar(),
            "safe_force_band_count": zero_scalar(),
            "hard_safety_count": zero_scalar(),
            "hard_force_count": zero_scalar(),
            "pushing_count": zero_scalar(),
            "fall_count": zero_scalar(),
            "nonfinite_count": zero_scalar(),
            "action_saturation_count": zero_scalar(),
            "action_rate_sum": zero_scalar(),
            "maximum_action_rate": zero_scalar(),
            "surface_gap_sum": zero_scalar(),
            "minimum_surface_gap_m": torch.full(
                (), float("inf"), dtype=torch.float64, device=self.device
            ),
            "maximum_surface_gap_m": zero_scalar(),
            "maximum_root_tilt_deg": zero_scalar(),
            "maximum_arm_torque_ratio": zero_scalar(),
            "minimum_arm_joint_margin_rad": torch.full(
                (), float("inf"), dtype=torch.float64, device=self.device
            ),
        }

    def drain_training_interval(self) -> dict[str, float]:
        """Return and reset exact control-step aggregates for one PPO iteration."""

        values = self._training_interval
        sample_count = float(values["sample_count"].item())
        denominator = max(sample_count, 1.0)
        action_denominator = max(sample_count * 2.0, 1.0)
        result = {
            "sample_count": sample_count,
            "bilateral_contact_fraction": float(values["bilateral_contact_count"].item())
            / denominator,
            "bilateral_onset_fraction": float(values["bilateral_onset_count"].item()) / denominator,
            "bilateral_contact_onset_fraction": float(values["bilateral_onset_count"].item())
            / denominator,
            "verify_fraction": float(values["verify_count"].item()) / denominator,
            "hold_success_fraction": float(values["hold_count"].item()) / denominator,
            "contact_retention_fraction": float(values["contact_retention_count"].item())
            / denominator,
            "force_band_occupancy": float(values["safe_force_band_count"].item()) / denominator,
            "force_band_occupancy_fraction": float(values["safe_force_band_count"].item())
            / denominator,
            "hard_force_termination_fraction": float(values["hard_force_count"].item())
            / denominator,
            "hard_safety_violation_fraction": float(values["hard_safety_count"].item())
            / denominator,
            "pushing_fraction": float(values["pushing_count"].item()) / denominator,
            "fall_fraction": float(values["fall_count"].item()) / denominator,
            "nonfinite_fraction": float(values["nonfinite_count"].item()) / denominator,
            "action_saturation_fraction": float(values["action_saturation_count"].item())
            / action_denominator,
            "mean_action_rate": float(values["action_rate_sum"].item()) / action_denominator,
            "maximum_action_rate": float(values["maximum_action_rate"].item()),
            "mean_surface_gap_m": float(values["surface_gap_sum"].item()) / action_denominator,
            "minimum_surface_gap_m": float(values["minimum_surface_gap_m"].item()),
            "maximum_surface_gap_m": float(values["maximum_surface_gap_m"].item()),
            "maximum_root_tilt_deg": float(values["maximum_root_tilt_deg"].item()),
            "maximum_arm_torque_ratio": float(values["maximum_arm_torque_ratio"].item()),
            "minimum_arm_joint_margin_rad": float(values["minimum_arm_joint_margin_rad"].item()),
        }
        for name, value in values.items():
            if name in {"minimum_surface_gap_m", "minimum_arm_joint_margin_rad"}:
                value.fill_(float("inf"))
            else:
                value.zero_()
        return result

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
        self.last_history_step[ids] = int(self.env.common_step_counter)
        self.gap_history_initialized[ids] = False
        self.previous_gap_sum[ids] = 0.0
        self.reward_mode[ids] = MODE_APPROACH
        self.contact_acquire_started[ids] = False
        self.bilateral_onset_seen[ids] = False
        self.left_onset_event[ids] = False
        self.right_onset_event[ids] = False
        self.bilateral_onset_event[ids] = False
        self.gap_progress_value[ids] = 0.0
        self.episode_contact_retention_steps[ids] = 0
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
        desired_pos = torch.tensor(CONTACT_TARGETS_OBJECT, device=self.device).expand(
            self.num_envs, -1, -1
        )
        desired_quat = torch.tensor(DESIRED_PALM_QUATERNION_OBJECT_WXYZ, device=self.device).expand(
            self.num_envs, 2, -1
        )
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
        right_force = torch.linalg.vector_norm(
            self.right_sensor.data.force_matrix_w[:, 0, 0], dim=-1
        )
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
            torch.acos(
                torch.clamp(1.0 - 2.0 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2), -1.0, 1.0)
            )
        )
        arm_q = self.robot.data.joint_pos[:, self.arm_ids]
        margin = (
            torch.minimum(arm_q - self.arm_limits[..., 0], self.arm_limits[..., 1] - arm_q)
            .min(dim=-1)
            .values
        )
        torque_ratio = (
            (self.robot.data.applied_torque[:, self.arm_ids].abs() / self.arm_effort_limits)
            .max(dim=-1)
            .values
        )
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
        active = not bool(getattr(self.env, "_s2_03t_bootstrap_mode", False))
        history_step = int(self.env.common_step_counter)
        advance = self.last_history_step != history_step
        if not active:
            advance.zero_()
        self.left_onset_event.zero_()
        self.right_onset_event.zero_()
        self.bilateral_onset_event.zero_()
        self.gap_progress_value.zero_()

        new_onset = contacts & (self.contact_onset_step < 0) & advance[:, None]
        self.left_onset_event.copy_(new_onset[:, 0])
        self.right_onset_event.copy_(new_onset[:, 1])
        self.contact_onset_step[new_onset] = current_step[:, None].expand(-1, 2)[new_onset]
        new_bilateral = contacts.all(dim=-1) & ~self.bilateral_onset_seen & advance
        self.bilateral_onset_event.copy_(new_bilateral)
        self.bilateral_onset_seen |= new_bilateral
        self.contact_acquire_started |= new_onset.any(dim=-1)
        within_impulse = (self.contact_onset_step >= 0) & (
            current_step[:, None] - self.contact_onset_step < IMPULSE_WINDOW_STEPS
        )
        self.impulse += torch.where(
            advance[:, None] & within_impulse,
            forces * self.dt,
            torch.zeros_like(forces),
        )

        physical_masks = {
            "failure_nonfinite": ~finite,
            "failure_forbidden_non_palm_box_collision": forbidden,
            "failure_force_peak": forces.max(dim=-1).values > FORCE_PEAK_THRESHOLD_N,
            "failure_palm_impulse": (self.impulse > PALM_IMPULSE_THRESHOLD_NS).any(dim=-1),
            "failure_combined_impulse": self.impulse.sum(dim=-1) > COMBINED_IMPULSE_THRESHOLD_NS,
            "failure_force_rate": force_rate > FORCE_RATE_THRESHOLD_NPS,
            "failure_box_linear_speed": box_linear_speed > BOX_LINEAR_SPEED_THRESHOLD_MPS,
            "failure_box_angular_speed": box_angular_speed > BOX_ANGULAR_SPEED_THRESHOLD_RADPS,
            "failure_box_translation": box_translation > BOX_TRANSLATION_THRESHOLD_M,
            "failure_box_yaw_change": box_yaw_delta.abs() > BOX_YAW_THRESHOLD_RAD,
            "failure_base_excursion": base_excursion > BASE_EXCURSION_THRESHOLD_M,
            "failure_root_height": (self.robot.data.root_link_pos_w[:, 2] < ROOT_HEIGHT_MIN_M)
            | (self.robot.data.root_link_pos_w[:, 2] > ROOT_HEIGHT_MAX_M),
            "failure_root_tilt": root_tilt > ROOT_TILT_MAX_DEG,
            "failure_arm_joint_margin": margin < ARM_MARGIN_MIN_RAD,
            "failure_arm_torque": torque_ratio > ARM_TORQUE_RATIO_MAX,
        }
        safety_failure = torch.zeros_like(finite)
        for mask in physical_masks.values():
            safety_failure |= mask

        both = contacts.all(dim=-1)
        one = contacts.any(dim=-1) & ~both
        verify_progress_step = advance & ~self.contact_verified & both
        if active:
            verifying = ~self.contact_verified
            self.verify_count = torch.where(
                advance & verifying & both,
                self.verify_count + 1,
                torch.where(
                    advance & verifying, torch.zeros_like(self.verify_count), self.verify_count
                ),
            )
            newly_verified = verifying & (self.verify_count >= VERIFY_STEPS)
            self.contact_verified |= newly_verified
            holding = self.contact_verified & ~newly_verified
            safe_hold = holding & both & ~safety_failure
            self.hold_count = torch.where(advance & safe_hold, self.hold_count + 1, self.hold_count)
            self.loss_streak = torch.where(
                advance[:, None] & holding[:, None] & ~contacts,
                self.loss_streak + 1,
                torch.where(
                    advance[:, None] & holding[:, None],
                    torch.zeros_like(self.loss_streak),
                    self.loss_streak,
                ),
            )
            self.single_hand_count = torch.where(
                advance & verifying & one,
                self.single_hand_count + 1,
                torch.where(
                    advance & verifying,
                    torch.zeros_like(self.single_hand_count),
                    self.single_hand_count,
                ),
            )
            self.episode_contact_retention_steps = torch.where(
                advance & self.contact_verified & both,
                self.episode_contact_retention_steps + 1,
                self.episode_contact_retention_steps,
            )
            self.reward_mode = torch.where(
                self.contact_verified,
                torch.full_like(self.reward_mode, MODE_VERIFY_HOLD),
                torch.where(
                    self.contact_acquire_started,
                    torch.full_like(self.reward_mode, MODE_CONTACT_ACQUIRE),
                    torch.full_like(self.reward_mode, MODE_APPROACH),
                ),
            )
            gap_sum = gaps.sum(dim=-1)
            delta = self.previous_gap_sum - gap_sum
            delta = torch.where(
                delta.abs() < GAP_PROGRESS_DEADBAND_M, torch.zeros_like(delta), delta
            )
            delta = torch.clamp(delta, -GAP_PROGRESS_CLIP_M, GAP_PROGRESS_CLIP_M)
            approach_update = advance & (self.reward_mode == MODE_APPROACH)
            self.gap_progress_value = torch.where(
                approach_update & self.gap_history_initialized,
                delta,
                torch.zeros_like(delta),
            )
            self.previous_gap_sum = torch.where(advance, gap_sum, self.previous_gap_sum)
            self.gap_history_initialized |= advance
            self.last_history_step = torch.where(
                advance,
                torch.full_like(self.last_history_step, history_step),
                self.last_history_step,
            )
        else:
            safe_hold = torch.zeros_like(both)
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
            "gap_progress": self.gap_progress_value.clone(),
            "reward_mode": self.reward_mode.clone(),
            "reward_mode_one_hot": torch.nn.functional.one_hot(
                self.reward_mode, num_classes=MODE_COUNT
            ).to(dtype=gaps.dtype),
            "left_contact_onset_event": self.left_onset_event.clone(),
            "right_contact_onset_event": self.right_onset_event.clone(),
            "bilateral_contact_onset_event": self.bilateral_onset_event.clone(),
            "bilateral_verify_step": (verify_progress_step if active else torch.zeros_like(both)),
            "contact_loss": contact_loss,
            "single_hand_timeout": single_hand_timeout,
            "success": success,
            "attached_hold_step": safe_hold,
            **physical_masks,
        }
        if active:
            interval = self._training_interval
            pushing_failure = (
                physical_masks["failure_box_linear_speed"]
                | physical_masks["failure_box_angular_speed"]
                | physical_masks["failure_box_translation"]
                | physical_masks["failure_box_yaw_change"]
            )
            fall_failure = (
                physical_masks["failure_base_excursion"]
                | physical_masks["failure_root_height"]
                | physical_masks["failure_root_tilt"]
            )
            hard_safety_failure = (
                physical_masks["failure_nonfinite"]
                | physical_masks["failure_forbidden_non_palm_box_collision"]
                | physical_masks["failure_force_peak"]
                | physical_masks["failure_palm_impulse"]
                | physical_masks["failure_combined_impulse"]
                | physical_masks["failure_force_rate"]
                | fall_failure
                | physical_masks["failure_arm_joint_margin"]
                | physical_masks["failure_arm_torque"]
            )
            hard_force_failure = (
                physical_masks["failure_force_peak"]
                | physical_masks["failure_palm_impulse"]
                | physical_masks["failure_combined_impulse"]
                | physical_masks["failure_force_rate"]
            )
            safe_force_band = (
                (forces >= CONTACT_FORCE_THRESHOLD_N) & (forces <= SOFT_FORCE_MAXIMUM_N)
            ).all(dim=-1)
            interval["sample_count"].add_(advance.sum())
            interval["bilateral_contact_count"].add_((advance & both).sum())
            interval["bilateral_onset_count"].add_(self.bilateral_onset_event.sum())
            interval["verify_count"].add_(verify_progress_step.sum())
            interval["hold_count"].add_((advance & safe_hold).sum())
            interval["contact_retention_count"].add_((advance & self.contact_verified & both).sum())
            interval["safe_force_band_count"].add_((advance & safe_force_band).sum())
            interval["hard_safety_count"].add_((advance & hard_safety_failure).sum())
            interval["hard_force_count"].add_((advance & hard_force_failure).sum())
            interval["pushing_count"].add_((advance & pushing_failure).sum())
            interval["fall_count"].add_((advance & fall_failure).sum())
            interval["nonfinite_count"].add_((advance & ~finite).sum())
            interval["action_saturation_count"].add_(
                (advance[:, None] & (self.arm.raw_actions.abs() >= 0.999)).sum()
            )
            interval["action_rate_sum"].add_((advance[:, None] * self.arm.action_delta.abs()).sum())
            interval["maximum_action_rate"].copy_(
                torch.maximum(
                    interval["maximum_action_rate"],
                    torch.where(
                        advance[:, None],
                        self.arm.action_delta.abs(),
                        torch.zeros_like(self.arm.action_delta),
                    ).max(),
                )
            )
            interval["surface_gap_sum"].add_((advance[:, None] * gaps).sum())
            interval["minimum_surface_gap_m"].copy_(
                torch.minimum(
                    interval["minimum_surface_gap_m"],
                    torch.where(advance[:, None], gaps, torch.full_like(gaps, float("inf"))).min(),
                )
            )
            interval["maximum_surface_gap_m"].copy_(
                torch.maximum(
                    interval["maximum_surface_gap_m"],
                    torch.where(advance[:, None], gaps, torch.zeros_like(gaps)).max(),
                )
            )
            interval["maximum_root_tilt_deg"].copy_(
                torch.maximum(
                    interval["maximum_root_tilt_deg"],
                    torch.where(advance, root_tilt, torch.zeros_like(root_tilt)).max(),
                )
            )
            interval["maximum_arm_torque_ratio"].copy_(
                torch.maximum(
                    interval["maximum_arm_torque_ratio"],
                    torch.where(advance, torque_ratio, torch.zeros_like(torque_ratio)).max(),
                )
            )
            interval["minimum_arm_joint_margin_rad"].copy_(
                torch.minimum(
                    interval["minimum_arm_joint_margin_rad"],
                    torch.where(advance, margin, torch.full_like(margin, float("inf"))).min(),
                )
            )
            self.episode_min_gap = torch.minimum(self.episode_min_gap, gaps)
            self.episode_max_force = torch.maximum(self.episode_max_force, forces)
            self.episode_max_box_translation = torch.maximum(
                self.episode_max_box_translation, box_translation
            )
            self.episode_max_box_yaw = torch.maximum(self.episode_max_box_yaw, box_yaw_delta.abs())
            self.episode_min_root_height = torch.minimum(
                self.episode_min_root_height, self.metrics["root_height"]
            )
            self.episode_max_root_tilt = torch.maximum(self.episode_max_root_tilt, root_tilt)
            self.episode_max_torque_ratio = torch.maximum(
                self.episode_max_torque_ratio, torque_ratio
            )
            self.episode_min_joint_margin = torch.minimum(self.episode_min_joint_margin, margin)
            self.episode_contact_seen |= both
        if active:
            self.previous_force[advance] = forces[advance]
        return self.metrics

    def _forbidden_overlap_mask(self) -> torch.Tensor:
        """Use the frozen PhysX overlap-box backend, gated by filtered contact activity."""

        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        matrix = self.robot_box_sensor.data.force_matrix_w
        active = filtered_contact_activity(matrix, self.num_envs)
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

            def callback(
                hit: Any,
                robot_prefix: str = robot_prefix,
                allowed: set[str] = allowed,
            ) -> bool:
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
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                    float(quaternion[0]),
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
                for name in sorted(TERMINATION_METRIC_NAMES)
                if bool(
                    metrics[
                        name if name in COUNTER_TERMINATION_METRIC_NAMES else f"failure_{name}"
                    ][env_id]
                )
            ]
            snapshots[env_id] = {
                "episode_steps": int(self.env.episode_length_buf[env_id]),
                "success": bool(metrics["success"][env_id]),
                "time_out": bool(self.env.reset_time_outs[env_id])
                if hasattr(self.env, "reset_time_outs")
                else False,
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
                "contact_retention_steps": int(self.episode_contact_retention_steps[env_id]),
                "curriculum_level": int(getattr(self.env, "_s2_03t_curriculum_level", 0)),
                "finite": bool(metrics["finite"][env_id]),
            }
        return snapshots


COUNTER_TERMINATION_METRIC_NAMES = {"contact_loss", "single_hand_timeout"}


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


def actor_frame_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Deployable 45D actor frame; exact simulator safety signals remain critic-only."""

    state = runtime_state(env)
    metric = state.ensure()
    robot = state.robot
    arm = state.arm

    components = (
        torch.clamp(robot.data.root_ang_vel_b, -5.0, 5.0),
        torch.clamp(robot.data.projected_gravity_b, -1.0, 1.0),
        torch.clamp(robot.data.joint_pos[:, state.arm_ids] - arm.reference, -1.0, 1.0),
        torch.clamp(robot.data.joint_vel[:, state.arm_ids], -10.0, 10.0),
        torch.clamp(metric["gaps"], -0.02, 0.10),
        arm.contact_latched.to(dtype=robot.data.joint_pos.dtype),
        torch.clamp(arm.nominal_displacement_m, 0.0, 0.09),
        metric["reward_mode_one_hot"],
        torch.clamp(arm.raw_actions, -1.0, 1.0),
    )
    observation = torch.cat(components, dim=-1)
    if tuple(observation.shape) != (env.num_envs, 45):
        raise RuntimeError(f"S2-03T actor frame shape mismatch: {tuple(observation.shape)}")
    if not bool(torch.isfinite(observation).all()):
        raise RuntimeError("S2-03T actor observation contains NaN/Inf")
    return observation


def policy_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compatibility alias for the v2 deployable actor frame."""

    return actor_frame_observation(env)


def critic_privileged_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """40D simulation-only critic append; never exported as actor input."""

    state = runtime_state(env)
    metric = state.ensure()
    arm = state.arm
    level = int(getattr(env, "_s2_03t_curriculum_level", 0))
    level_tensor = torch.full((env.num_envs,), level, dtype=torch.long, device=env.device)
    curriculum_one_hot = torch.nn.functional.one_hot(level_tensor, num_classes=4).to(
        dtype=metric["gaps"].dtype
    )
    components = (
        torch.clamp(metric["forces"] / FORCE_PEAK_THRESHOLD_N, 0.0, 2.0),
        torch.clamp((metric["force_rate"] / FORCE_RATE_THRESHOLD_NPS).unsqueeze(-1), 0.0, 2.0),
        torch.clamp(
            metric["forces"].new_tensor(1.0) * state.impulse / PALM_IMPULSE_THRESHOLD_NS, 0.0, 2.0
        ),
        torch.clamp(metric["palm_position_error"], -0.10, 0.10).reshape(env.num_envs, 6),
        torch.clamp(metric["palm_orientation_error"], -0.5, 0.5).reshape(env.num_envs, 6),
        torch.clamp(
            metric["box_translation_vector"],
            -BOX_TRANSLATION_THRESHOLD_M,
            BOX_TRANSLATION_THRESHOLD_M,
        ),
        torch.clamp(
            metric["box_yaw_change"].unsqueeze(-1), -BOX_YAW_THRESHOLD_RAD, BOX_YAW_THRESHOLD_RAD
        ),
        torch.clamp(
            metric["box_linear_velocity"],
            -BOX_LINEAR_SPEED_THRESHOLD_MPS,
            BOX_LINEAR_SPEED_THRESHOLD_MPS,
        ),
        torch.clamp(
            metric["box_angular_velocity"],
            -BOX_ANGULAR_SPEED_THRESHOLD_RADPS,
            BOX_ANGULAR_SPEED_THRESHOLD_RADPS,
        ),
        torch.clamp(metric["base_excursion"].unsqueeze(-1), 0.0, BASE_EXCURSION_THRESHOLD_M),
        torch.clamp(metric["root_height"].unsqueeze(-1), ROOT_HEIGHT_MIN_M, ROOT_HEIGHT_MAX_M),
        torch.clamp(metric["root_tilt_deg"].unsqueeze(-1), 0.0, ROOT_TILT_MAX_DEG),
        torch.clamp(metric["arm_joint_margin"].unsqueeze(-1), 0.0, 1.0),
        torch.clamp(metric["arm_torque_ratio"].unsqueeze(-1), 0.0, 2.0),
        torch.clamp(arm.normal_jacobian_authority, 0.0, 10.0),
        arm.joint_target_clamped.to(dtype=metric["gaps"].dtype),
        curriculum_one_hot,
    )
    observation = torch.cat(components, dim=-1)
    if tuple(observation.shape) != (env.num_envs, 40):
        raise RuntimeError(f"S2-03T critic append shape mismatch: {tuple(observation.shape)}")
    if not bool(torch.isfinite(observation).all()):
        raise RuntimeError("S2-03T critic observation contains NaN/Inf")
    return observation


def _mode_mask(metric: dict[str, torch.Tensor], mode: int) -> torch.Tensor:
    return metric["reward_mode"] == mode


# Reward formulas. The config owns the frozen v2 weights; RewardManager applies dt.
def gap_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    return torch.where(
        _mode_mask(metric, MODE_APPROACH),
        metric["gap_progress"],
        torch.zeros_like(metric["gap_progress"]),
    )


def approach_symmetry(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    penalty = -torch.clamp(torch.abs(metric["gaps"][:, 0] - metric["gaps"][:, 1]), max=0.001)
    return torch.where(_mode_mask(metric, MODE_APPROACH), penalty, torch.zeros_like(penalty))


def left_contact_onset(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["left_contact_onset_event"].float()


def right_contact_onset(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["right_contact_onset_event"].float()


def bilateral_contact_onset(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["bilateral_contact_onset_event"].float()


def verify_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["bilateral_verify_step"].float() / float(VERIFY_STEPS)


def safe_force_band(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    in_band = (
        (metric["forces"] >= CONTACT_FORCE_THRESHOLD_N) & (metric["forces"] <= SOFT_FORCE_MAXIMUM_N)
    ).all(dim=-1)
    enabled = _mode_mask(metric, MODE_CONTACT_ACQUIRE)
    return (in_band & enabled).float()


def single_contact_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    single = metric["contacts"].any(dim=-1) & ~metric["contacts"].all(dim=-1)
    return (single & _mode_mask(metric, MODE_CONTACT_ACQUIRE)).float()


def contact_retention(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    return (metric["contacts"].all(dim=-1) & _mode_mask(metric, MODE_VERIFY_HOLD)).float()


def attached_hold_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["attached_hold_step"].float()


def force_balance(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    span = SOFT_FORCE_MAXIMUM_N - CONTACT_FORCE_THRESHOLD_N
    balance = torch.clamp(
        1.0 - torch.abs(metric["forces"][:, 0] - metric["forces"][:, 1]) / span, 0.0, 1.0
    )
    enabled = metric["contacts"].all(dim=-1) & _mode_mask(metric, MODE_VERIFY_HOLD)
    return torch.where(enabled, balance, torch.zeros_like(balance))


def success_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["success"].float()


def time_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    if bool(getattr(env, "_s2_03t_bootstrap_mode", False)):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.ones(env.num_envs, device=env.device)


def hand_orientation_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.linalg.vector_norm(
        runtime_state(env).ensure()["palm_orientation_error"], dim=-1
    ).sum(dim=-1)


def unsafe_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    scale = FORCE_PEAK_THRESHOLD_N - SOFT_FORCE_MAXIMUM_N
    excess = torch.clamp((metric["forces"] - SOFT_FORCE_MAXIMUM_N) / scale, 0.0, 1.0)
    enabled = metric["reward_mode"] != MODE_APPROACH
    return torch.where(
        enabled, torch.square(excess).sum(dim=-1), torch.zeros(env.num_envs, device=env.device)
    )


def force_impulse_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = runtime_state(env)
    metric = state.ensure()
    value = (
        metric["forces"].max(dim=-1).values / FORCE_PEAK_THRESHOLD_N
        + state.impulse.sum(dim=-1) / COMBINED_IMPULSE_THRESHOLD_NS
        + metric["force_rate"] / FORCE_RATE_THRESHOLD_NPS
    )
    return torch.where(metric["reward_mode"] != MODE_APPROACH, value, torch.zeros_like(value))


def force_imbalance(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    value = torch.abs(metric["forces"][:, 0] - metric["forces"][:, 1])
    return torch.where(_mode_mask(metric, MODE_VERIFY_HOLD), value, torch.zeros_like(value))


def box_translation(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["box_translation"]


def box_yaw_change(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["box_yaw_change"].abs()


def root_risk(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(
        runtime_state(env).ensure()["root_tilt_deg"] - ROOT_RISK_DEADBAND_DEG, min=0.0
    )


def joint_limit_margin(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(
        ARM_MARGIN_MIN_RAD - runtime_state(env).ensure()["arm_joint_margin"], min=0.0
    )


def torque_ratio(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(runtime_state(env).ensure()["arm_torque_ratio"] - 1.0, min=0.0)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(runtime_state(env).arm.raw_actions).sum(dim=-1)


def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(runtime_state(env).arm.action_delta).sum(dim=-1)


def hard_force_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    return (
        metric["failure_force_peak"]
        | metric["failure_palm_impulse"]
        | metric["failure_combined_impulse"]
        | metric["failure_force_rate"]
    ).float()


def pushing_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    metric = runtime_state(env).ensure()
    return (
        metric["failure_box_linear_speed"]
        | metric["failure_box_angular_speed"]
        | metric["failure_box_translation"]
        | metric["failure_box_yaw_change"]
    ).float()


def contact_loss_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["contact_loss"].float()


def forbidden_collision_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    return runtime_state(env).ensure()["failure_forbidden_non_palm_box_collision"].float()


def contact_timeout_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    return isaac_time_out(env).float()


def contact_gap_curriculum(
    env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor
) -> dict[str, float]:
    return env.update_contact_curriculum(env_ids)


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    if getattr(env, "_s2_03t_bootstrap_mode", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return runtime_state(env).ensure()["success"]


def metric_termination(env: ManagerBasedRLEnv, metric_name: str) -> torch.Tensor:
    if getattr(env, "_s2_03t_bootstrap_mode", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    metrics = runtime_state(env).ensure()
    key = (
        metric_name if metric_name in COUNTER_TERMINATION_METRIC_NAMES else f"failure_{metric_name}"
    )
    return metrics[key]


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    if getattr(env, "_s2_03t_bootstrap_mode", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return isaac_time_out(env)
