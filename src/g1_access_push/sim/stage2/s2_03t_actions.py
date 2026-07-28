"""S2-03T hybrid palm-normal and frozen batched lower-body action terms.

The public policy action is exactly two normalized palm-normal corrections.
A deterministic, jerk-limited nominal trajectory supplies the reach authority
that the former 14-D joint residual did not have.  The certified locomotion
student remains internal and contributes zero dimensions to the policy action
space.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import retrieve_file_path

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
LEFT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[0::2]
RIGHT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[1::2]

BASE_COMMAND = (0.0, 0.0, 0.0, 0.7)
BASE_COMMAND_DIM = 4
STUDENT_OBSERVATION_DIM = 64
PREVIOUS_LOWER_ACTION_DIM = 12
STUDENT_INPUT_DIM = 80
STUDENT_OUTPUT_DIM = 12
RECURRENT_STATE_DIM = 256
HYBRID_ACTION_DIM = 2
HYBRID_ACTION_ORDER = (
    "left_palm_normal_correction",
    "right_palm_normal_correction",
)
PALM_BODY_NAMES = ("left_hand_palm_link", "right_hand_palm_link")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batched_student_forward(
    policy: torch.jit.ScriptModule,
    policy_input: torch.Tensor,
    hidden_state: torch.Tensor,
    cell_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exported student with external per-environment recurrent state."""

    if policy_input.ndim != 2 or policy_input.shape[1] != STUDENT_INPUT_DIM:
        raise ValueError(
            f"policy_input must be [N,{STUDENT_INPUT_DIM}], got {tuple(policy_input.shape)}"
        )
    expected_state = (1, policy_input.shape[0], RECURRENT_STATE_DIM)
    if tuple(hidden_state.shape) != expected_state or tuple(cell_state.shape) != expected_state:
        raise ValueError(f"recurrent state must be {expected_state}")
    normalized = policy.normalizer(policy_input.unsqueeze(0))
    rnn_output, (next_hidden, next_cell) = policy.rnn.forward__0(
        normalized, (hidden_state, cell_state)
    )
    output = policy.actor(rnn_output.squeeze(0))
    if tuple(output.shape) != (policy_input.shape[0], STUDENT_OUTPUT_DIM):
        raise RuntimeError(f"student output shape mismatch: {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("certified recurrent student produced non-finite output")
    return output, next_hidden, next_cell


class HybridNormalApproachAction(ActionTerm):
    """Deterministic normal approach plus learned bilateral normal correction.

    Reset-time box and palm poses remain the target anchor for the episode.
    Nominal trajectory and correction state advance only in process_actions;
    apply_actions only solves DLS inside a fixed control-step joint envelope.
    """

    cfg: HybridNormalApproachActionCfg
    _asset: Articulation

    def __init__(self, cfg: HybridNormalApproachActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            list(cfg.joint_names), preserve_order=True
        )
        if tuple(self._joint_names) != ARM_JOINT_NAMES:
            raise RuntimeError(f"arm joint order mismatch: {self._joint_names}")
        self._left_joint_ids, left_names = self._asset.find_joints(
            list(LEFT_ARM_JOINT_NAMES), preserve_order=True
        )
        self._right_joint_ids, right_names = self._asset.find_joints(
            list(RIGHT_ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(left_names) != LEFT_ARM_JOINT_NAMES or tuple(right_names) != RIGHT_ARM_JOINT_NAMES:
            raise RuntimeError("left/right IK joint order mismatch")

        body_ids, body_names = self._asset.find_bodies(list(PALM_BODY_NAMES), preserve_order=True)
        if tuple(body_names) != PALM_BODY_NAMES:
            raise RuntimeError(f"palm body order mismatch: {body_names}")
        self._body_ids = body_ids
        self._jacobian_body_ids = (
            body_ids if not self._asset.is_fixed_base else [value - 1 for value in body_ids]
        )
        self._left_jacobian_joint_ids = [value + 6 for value in self._left_joint_ids]
        self._right_jacobian_joint_ids = [value + 6 for value in self._right_joint_ids]

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": float(cfg.dls_damping_lambda)},
        )
        self._left_ik = DifferentialIKController(ik_cfg, self.num_envs, self.device)
        self._right_ik = DifferentialIKController(ik_cfg, self.num_envs, self.device)

        action_shape = (self.num_envs, HYBRID_ACTION_DIM)
        self._raw_actions = torch.zeros(action_shape, device=self.device)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._action_delta = torch.zeros_like(self._raw_actions)
        self._correction_m = torch.zeros_like(self._raw_actions)
        self._contact_correction_at_latch = torch.zeros_like(self._raw_actions)
        self._nominal_displacement_m = torch.zeros_like(self._raw_actions)
        self._nominal_velocity_mps = torch.zeros_like(self._raw_actions)
        self._nominal_acceleration_mps2 = torch.zeros_like(self._raw_actions)
        self._contact_latched = torch.zeros(action_shape, dtype=torch.bool, device=self.device)
        self._safety_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._contact_onset_age = torch.full(action_shape, -1, dtype=torch.long, device=self.device)
        self._contact_impulse_ns = torch.zeros_like(self._raw_actions)
        self._previous_force_n = torch.zeros_like(self._raw_actions)
        self._reference = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._reference.clone()
        self._control_step_joint_origin = self._processed_actions.clone()
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._target_lower = limits[..., 0] + float(cfg.joint_limit_margin_rad)
        self._target_upper = limits[..., 1] - float(cfg.joint_limit_margin_rad)
        if bool((self._target_lower >= self._target_upper).any()):
            raise RuntimeError("arm joint limits cannot preserve the frozen margin")

        self._box = env.scene[cfg.box_asset_name]
        self._left_contact_sensor = env.scene[cfg.left_contact_sensor_name]
        self._right_contact_sensor = env.scene[cfg.right_contact_sensor_name]
        self._forbidden_contact_sensor = env.scene[cfg.forbidden_contact_sensor_name]
        filter_expressions = self._forbidden_contact_sensor.cfg.filter_prim_paths_expr
        self._allowed_forbidden_filter_indices = tuple(
            index
            for index, expression in enumerate(filter_expressions)
            if str(expression).endswith(PALM_BODY_NAMES)
        )
        if len(self._allowed_forbidden_filter_indices) != 2:
            raise RuntimeError("PALM_CONTACT_FILTER_WHITELIST_INVALID")
        self._arm_effort_limits = (
            self._asset.data.joint_effort_limits[:, self._joint_ids].abs().clamp_min(1.0e-6)
        )
        self._anchor_box_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._anchor_box_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._anchor_box_quat_w[:, 0] = 1.0
        self._anchor_box_yaw = torch.zeros(self.num_envs, device=self.device)
        self._anchor_base_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._reference_palm_pos_object = torch.zeros((self.num_envs, 2, 3), device=self.device)
        self._anchor_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_pose_b = torch.zeros((self.num_envs, 2, 7), device=self.device)
        self._target_pose_b[..., 3] = 1.0
        self._normal_jacobian_authority = torch.zeros(action_shape, device=self.device)
        self._joint_target_clamped = torch.zeros(action_shape, dtype=torch.bool, device=self.device)
        self._bootstrap_mode = False
        self._bootstrap_commands = torch.zeros((self.num_envs, 2, 6), device=self.device)
        self._bootstrap_targets = torch.zeros_like(self._processed_actions)

    @property
    def action_dim(self) -> int:
        return HYBRID_ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def previous_raw_actions(self) -> torch.Tensor:
        return self._previous_raw_actions

    @property
    def action_delta(self) -> torch.Tensor:
        return self._action_delta

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def applied_residual(self) -> torch.Tensor:
        """Compatibility view of the applied 14-D joint offset in radians."""

        return self._processed_actions - self._reference

    @property
    def correction_m(self) -> torch.Tensor:
        return self._correction_m

    @property
    def nominal_displacement_m(self) -> torch.Tensor:
        return self._nominal_displacement_m

    @property
    def nominal_velocity_mps(self) -> torch.Tensor:
        return self._nominal_velocity_mps

    @property
    def contact_latched(self) -> torch.Tensor:
        return self._contact_latched

    @property
    def safety_latched(self) -> torch.Tensor:
        return self._safety_latched

    @property
    def contact_impulse_ns(self) -> torch.Tensor:
        return self._contact_impulse_ns

    @property
    def normal_jacobian_authority(self) -> torch.Tensor:
        return self._normal_jacobian_authority

    @property
    def joint_target_clamped(self) -> torch.Tensor:
        return self._joint_target_clamped

    @property
    def target_pose_b(self) -> torch.Tensor:
        return self._target_pose_b

    @property
    def anchor_box_pose_w(self) -> torch.Tensor:
        return torch.cat((self._anchor_box_pos_w, self._anchor_box_quat_w), dim=-1)

    @property
    def reference(self) -> torch.Tensor:
        return self._reference

    @property
    def bootstrap_targets(self) -> torch.Tensor:
        return self._bootstrap_targets

    @property
    def joint_ids(self) -> list[int]:
        return list(self._joint_ids)

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    def set_reference(self, reference: torch.Tensor) -> None:
        reference = reference.to(device=self.device, dtype=self._reference.dtype)
        if reference.ndim == 1:
            reference = reference.unsqueeze(0)
        if reference.shape == (1, len(ARM_JOINT_NAMES)):
            reference = reference.expand(self.num_envs, -1)
        if tuple(reference.shape) != tuple(self._reference.shape):
            raise ValueError(f"arm reference shape mismatch: {tuple(reference.shape)}")
        if not bool(torch.isfinite(reference).all()):
            raise ValueError("arm reference must be finite")
        if bool(((reference < self._target_lower) | (reference > self._target_upper)).any()):
            raise ValueError("arm reference violates the frozen 0.10 rad joint margin")
        self._reference.copy_(reference)
        self._processed_actions.copy_(reference)
        self._control_step_joint_origin.copy_(reference)
        self.reset()

    def restore_reference(self, env_ids: torch.Tensor, reference: torch.Tensor) -> None:
        """Restore a subset without perturbing recurrent episodes in other envs."""

        reference = reference.to(device=self.device, dtype=self._reference.dtype)
        if reference.ndim == 1:
            reference = reference.unsqueeze(0)
        if reference.shape[0] == 1:
            reference = reference.expand(len(env_ids), -1)
        if tuple(reference.shape) != (len(env_ids), len(ARM_JOINT_NAMES)):
            raise ValueError(f"arm subset reference shape mismatch: {tuple(reference.shape)}")
        if not bool(torch.isfinite(reference).all()):
            raise ValueError("arm subset reference must be finite")
        if bool(
            (
                (reference < self._target_lower[env_ids])
                | (reference > self._target_upper[env_ids])
            ).any()
        ):
            raise ValueError("arm subset reference violates the frozen 0.10 rad joint margin")
        self._reference[env_ids] = reference
        self.reset(env_ids)

    def set_episode_anchor(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        box_pos_w: torch.Tensor | None = None,
        box_quat_w: torch.Tensor | None = None,
    ) -> None:
        """Freeze reset-time box/palm poses used by selected environments."""

        ids = self._resolve_env_ids(env_ids)
        count = len(ids)
        if count == 0:
            return
        if box_pos_w is None:
            box_pos_w = self._box.data.root_link_pos_w[ids]
        if box_quat_w is None:
            box_quat_w = self._box.data.root_link_quat_w[ids]
        box_pos_w = self._expand_rows(box_pos_w, count, 3, "box anchor position")
        box_quat_w = self._expand_rows(box_quat_w, count, 4, "box anchor quaternion")
        if not bool(torch.isfinite(box_pos_w).all() and torch.isfinite(box_quat_w).all()):
            raise ValueError("box anchor pose must be finite")
        palm_pos_w = self._asset.data.body_pos_w[ids][:, self._body_ids].reshape(-1, 3)
        palm_quat_w = self._asset.data.body_quat_w[ids][:, self._body_ids].reshape(-1, 4)
        palm_pos_object, _ = math_utils.subtract_frame_transforms(
            box_pos_w[:, None, :].expand(-1, 2, -1).reshape(-1, 3),
            box_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
            palm_pos_w,
            palm_quat_w,
        )
        self._anchor_box_pos_w[ids] = box_pos_w
        self._anchor_box_quat_w[ids] = box_quat_w
        self._anchor_box_yaw[ids] = self._quat_yaw(box_quat_w)
        self._anchor_base_xy[ids] = self._asset.data.root_link_pos_w[ids, :2]
        self._reference_palm_pos_object[ids] = palm_pos_object.reshape(count, 2, 3)
        self._anchor_valid[ids] = True

    def set_bootstrap_mode(self, enabled: bool) -> None:
        self._bootstrap_mode = bool(enabled)

    def set_bootstrap_commands(self, commands: torch.Tensor) -> None:
        if tuple(commands.shape) != (self.num_envs, 2, 6):
            raise ValueError(f"bootstrap commands must be [N,2,6], got {tuple(commands.shape)}")
        self._bootstrap_commands.copy_(commands)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        ids = self._resolve_env_ids(env_ids)
        self._raw_actions[ids] = 0.0
        self._previous_raw_actions[ids] = 0.0
        self._action_delta[ids] = 0.0
        self._correction_m[ids] = 0.0
        self._contact_correction_at_latch[ids] = 0.0
        self._nominal_displacement_m[ids] = 0.0
        self._nominal_velocity_mps[ids] = 0.0
        self._nominal_acceleration_mps2[ids] = 0.0
        self._contact_latched[ids] = False
        self._safety_latched[ids] = False
        self._contact_onset_age[ids] = -1
        self._contact_impulse_ns[ids] = 0.0
        self._previous_force_n[ids] = 0.0
        self._processed_actions[ids] = self._reference[ids]
        self._control_step_joint_origin[ids] = self._reference[ids]
        self._bootstrap_commands[ids] = 0.0
        self._bootstrap_targets[ids] = self._reference[ids]
        self._normal_jacobian_authority[ids] = 0.0
        self._joint_target_clamped[ids] = False
        self._anchor_valid[ids] = False

    def process_actions(self, actions: torch.Tensor) -> None:
        if tuple(actions.shape) != tuple(self._raw_actions.shape):
            raise ValueError(f"arm action shape mismatch: {tuple(actions.shape)}")
        if not bool(torch.isfinite(actions).all()):
            raise RuntimeError("non-finite arm action")
        clipped = torch.clamp(actions, -1.0, 1.0)
        self._previous_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(clipped)
        self._action_delta.copy_(self._raw_actions - self._previous_raw_actions)
        if self._bootstrap_mode:
            left_pos, left_quat = self._compute_frame_pose(0)
            right_pos, right_quat = self._compute_frame_pose(1)
            self._left_ik.set_command(self._bootstrap_commands[:, 0], left_pos, left_quat)
            self._right_ik.set_command(self._bootstrap_commands[:, 1], right_pos, right_quat)
            return
        missing_anchor = (~self._anchor_valid).nonzero(as_tuple=False).squeeze(-1)
        if len(missing_anchor) > 0:
            self.set_episode_anchor(missing_anchor)
        forces, contacts = self._contact_state()
        newly_contacted = contacts & ~self._contact_latched
        self._contact_correction_at_latch[newly_contacted] = self._correction_m[newly_contacted]
        self._contact_latched |= contacts
        self._update_safety_latch(forces, newly_contacted)

        active = ~self._safety_latched[:, None]
        desired_correction = clipped * float(self.cfg.correction_scale_m)
        contacted_inward_cap = self._contact_correction_at_latch + float(
            self.cfg.contacted_hand_additional_inward_cap_m
        )
        desired_correction = torch.where(
            self._contact_latched,
            torch.minimum(desired_correction, contacted_inward_cap),
            desired_correction,
        )
        correction_delta = torch.clamp(
            desired_correction - self._correction_m,
            -float(self.cfg.correction_rate_limit_m_per_control_step),
            float(self.cfg.correction_rate_limit_m_per_control_step),
        )
        self._correction_m.add_(
            torch.where(active, correction_delta, torch.zeros_like(correction_delta))
        )
        self._correction_m.clamp_(
            -float(self.cfg.correction_scale_m), float(self.cfg.correction_scale_m)
        )

        nominal_active = active & ~self._contact_latched
        dt = float(self.cfg.control_dt_s)
        desired_acceleration = torch.clamp(
            (float(self.cfg.nominal_speed_limit_mps) - self._nominal_velocity_mps) / dt,
            -float(self.cfg.nominal_acceleration_limit_mps2),
            float(self.cfg.nominal_acceleration_limit_mps2),
        )
        acceleration_delta = torch.clamp(
            desired_acceleration - self._nominal_acceleration_mps2,
            -float(self.cfg.nominal_jerk_limit_mps3) * dt,
            float(self.cfg.nominal_jerk_limit_mps3) * dt,
        )
        next_acceleration = self._nominal_acceleration_mps2 + acceleration_delta
        next_velocity = torch.clamp(
            self._nominal_velocity_mps + next_acceleration * dt,
            min=0.0,
            max=float(self.cfg.nominal_speed_limit_mps),
        )
        next_displacement = torch.clamp(
            self._nominal_displacement_m + next_velocity * dt,
            min=0.0,
            max=float(self.cfg.nominal_maximum_displacement_m),
        )
        self._nominal_acceleration_mps2.copy_(
            torch.where(nominal_active, next_acceleration, torch.zeros_like(next_acceleration))
        )
        self._nominal_velocity_mps.copy_(
            torch.where(nominal_active, next_velocity, torch.zeros_like(next_velocity))
        )
        self._nominal_displacement_m.copy_(
            torch.where(nominal_active, next_displacement, self._nominal_displacement_m)
        )
        self._control_step_joint_origin.copy_(self._processed_actions)
        self._joint_target_clamped.zero_()
        self._set_hybrid_ik_commands()

    def apply_actions(self) -> None:
        left_pos, left_quat = self._compute_frame_pose(0)
        right_pos, right_quat = self._compute_frame_pose(1)
        left_q = self._asset.data.joint_pos[:, self._left_joint_ids]
        right_q = self._asset.data.joint_pos[:, self._right_joint_ids]
        left_target = self._left_ik.compute(
            left_pos, left_quat, self._compute_frame_jacobian(0), left_q
        )
        right_target = self._right_ik.compute(
            right_pos, right_quat, self._compute_frame_jacobian(1), right_q
        )
        if not self._bootstrap_mode:
            self._apply_hybrid_joint_target(0, left_target)
            self._apply_hybrid_joint_target(1, right_target)
            self._processed_actions.copy_(
                torch.where(
                    self._safety_latched[:, None],
                    self._control_step_joint_origin,
                    self._processed_actions,
                )
            )
            self._asset.set_joint_position_target(
                self._processed_actions, joint_ids=self._joint_ids
            )
            return
        self._asset.set_joint_position_target(left_target, joint_ids=self._left_joint_ids)
        self._asset.set_joint_position_target(right_target, joint_ids=self._right_joint_ids)
        self._bootstrap_targets[:, 0::2] = left_target
        self._bootstrap_targets[:, 1::2] = right_target

    def _set_hybrid_ik_commands(self) -> None:
        displacement = self._nominal_displacement_m + self._correction_m
        target_pos_object = self._reference_palm_pos_object.clone()
        target_pos_object[..., 0] += displacement
        target_quat_object = torch.tensor(
            self.cfg.palm_orientation_object_wxyz,
            device=self.device,
            dtype=target_pos_object.dtype,
        ).expand(self.num_envs, 2, -1)
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            self._anchor_box_pos_w[:, None, :].expand(-1, 2, -1).reshape(-1, 3),
            self._anchor_box_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
            target_pos_object.reshape(-1, 3),
            target_quat_object.reshape(-1, 4),
        )
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w[:, None, :].expand(-1, 2, -1).reshape(-1, 3),
            self._asset.data.root_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
            target_pos_w,
            target_quat_w,
        )
        self._target_pose_b[..., :3] = target_pos_b.reshape(self.num_envs, 2, 3)
        self._target_pose_b[..., 3:] = target_quat_b.reshape(self.num_envs, 2, 4)
        normal_w = math_utils.matrix_from_quat(self._anchor_box_quat_w)[:, :, 0]
        normal_b = torch.bmm(
            math_utils.matrix_from_quat(math_utils.quat_inv(self._asset.data.root_quat_w)),
            normal_w.unsqueeze(-1),
        ).squeeze(-1)
        for side, controller in enumerate((self._left_ik, self._right_ik)):
            current_pos, current_quat = self._compute_frame_pose(side)
            position_error, orientation_error = math_utils.compute_pose_error(
                current_pos,
                current_quat,
                self._target_pose_b[:, side, :3],
                self._target_pose_b[:, side, 3:],
                rot_error_type="axis_angle",
            )
            command = torch.cat((position_error, orientation_error), dim=-1)
            command_finite = torch.isfinite(command).all(dim=-1)
            self._safety_latched |= ~command_finite
            command = torch.where(command_finite[:, None], command, torch.zeros_like(command))
            controller.set_command(command, current_pos, current_quat)
            jacobian = self._compute_frame_jacobian(side)
            normal_row = torch.bmm(normal_b.unsqueeze(1), jacobian[:, :3]).squeeze(1)
            self._normal_jacobian_authority[:, side] = torch.linalg.vector_norm(normal_row, dim=-1)

    def _apply_hybrid_joint_target(self, side: int, unconstrained: torch.Tensor) -> None:
        target_slice = slice(side, len(ARM_JOINT_NAMES), 2)
        current_target = self._processed_actions[:, target_slice]
        finite = torch.isfinite(unconstrained).all(dim=-1)
        self._safety_latched |= ~finite
        safe_unconstrained = torch.where(finite[:, None], unconstrained, current_target)
        lower = torch.maximum(
            self._target_lower[:, target_slice],
            self._control_step_joint_origin[:, target_slice]
            - float(self.cfg.joint_target_rate_limit_rad_per_control_step),
        )
        upper = torch.minimum(
            self._target_upper[:, target_slice],
            self._control_step_joint_origin[:, target_slice]
            + float(self.cfg.joint_target_rate_limit_rad_per_control_step),
        )
        constrained = torch.maximum(torch.minimum(safe_unconstrained, upper), lower)
        constrained = torch.where(self._safety_latched[:, None], current_target, constrained)
        self._joint_target_clamped[:, side] |= (
            torch.abs(constrained - safe_unconstrained) > 1.0e-8
        ).any(dim=-1)
        self._processed_actions[:, target_slice] = constrained

    def _contact_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        force_values = []
        for sensor in (self._left_contact_sensor, self._right_contact_sensor):
            matrix = sensor.data.force_matrix_w
            if matrix is None or matrix.ndim != 4 or matrix.shape[0] != self.num_envs:
                raise RuntimeError("PALM_CONTACT_FORCE_MATRIX_INVALID")
            force_values.append(torch.linalg.vector_norm(matrix[:, 0, 0], dim=-1))
        forces = torch.stack(force_values, dim=-1)
        self._safety_latched |= ~torch.isfinite(forces).all(dim=-1)
        return forces, forces >= float(self.cfg.contact_force_threshold_n)

    def _update_safety_latch(self, forces: torch.Tensor, newly_contacted: torch.Tensor) -> None:
        self._contact_onset_age[newly_contacted] = 0
        within_impulse_window = (self._contact_onset_age >= 0) & (
            self._contact_onset_age < int(self.cfg.impulse_window_steps)
        )
        self._contact_impulse_ns.add_(
            torch.where(
                within_impulse_window,
                forces * float(self.cfg.control_dt_s),
                torch.zeros_like(forces),
            )
        )
        self._contact_onset_age = torch.where(
            self._contact_onset_age >= 0, self._contact_onset_age + 1, self._contact_onset_age
        )
        force_rate = torch.max(torch.abs(forces - self._previous_force_n), dim=-1).values / float(
            self.cfg.control_dt_s
        )
        self._previous_force_n.copy_(forces)
        box_translation = torch.linalg.vector_norm(
            self._box.data.root_link_pos_w - self._anchor_box_pos_w, dim=-1
        )
        current_box_yaw = self._quat_yaw(self._box.data.root_link_quat_w)
        box_yaw_delta = torch.atan2(
            torch.sin(current_box_yaw - self._anchor_box_yaw),
            torch.cos(current_box_yaw - self._anchor_box_yaw),
        ).abs()
        box_linear_speed = torch.linalg.vector_norm(self._box.data.root_com_lin_vel_w, dim=-1)
        box_angular_speed = torch.linalg.vector_norm(self._box.data.root_com_ang_vel_w, dim=-1)
        base_excursion = torch.linalg.vector_norm(
            self._asset.data.root_link_pos_w[:, :2] - self._anchor_base_xy, dim=-1
        )
        root_quat = self._asset.data.root_link_quat_w
        root_tilt_deg = torch.rad2deg(
            torch.acos(
                torch.clamp(
                    1.0 - 2.0 * (root_quat[:, 1].square() + root_quat[:, 2].square()),
                    -1.0,
                    1.0,
                )
            )
        )
        arm_q = self._asset.data.joint_pos[:, self._joint_ids]
        joint_margin = (
            torch.minimum(
                arm_q - self._asset.data.joint_pos_limits[:, self._joint_ids, 0],
                self._asset.data.joint_pos_limits[:, self._joint_ids, 1] - arm_q,
            )
            .min(dim=-1)
            .values
        )
        torque_ratio = (
            (self._asset.data.applied_torque[:, self._joint_ids].abs() / self._arm_effort_limits)
            .max(dim=-1)
            .values
        )
        forbidden = self._filtered_contact_active(
            self._forbidden_contact_sensor.data.force_matrix_w
        )
        finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for value in (
            forces,
            self._contact_impulse_ns,
            self._box.data.root_state_w,
            self._asset.data.root_state_w,
            self._asset.data.joint_pos,
            self._asset.data.applied_torque,
        ):
            finite &= torch.isfinite(value).reshape(self.num_envs, -1).all(dim=-1)
        failure = (
            ~finite
            | forbidden
            | (forces.max(dim=-1).values > float(self.cfg.force_peak_threshold_n))
            | (self._contact_impulse_ns > float(self.cfg.palm_impulse_threshold_ns)).any(dim=-1)
            | (self._contact_impulse_ns.sum(dim=-1) > float(self.cfg.combined_impulse_threshold_ns))
            | (force_rate > float(self.cfg.force_rate_threshold_nps))
            | (box_linear_speed > float(self.cfg.box_linear_speed_threshold_mps))
            | (box_angular_speed > float(self.cfg.box_angular_speed_threshold_radps))
            | (box_translation > float(self.cfg.box_translation_threshold_m))
            | (box_yaw_delta > float(self.cfg.box_yaw_threshold_rad))
            | (base_excursion > float(self.cfg.base_excursion_threshold_m))
            | (self._asset.data.root_link_pos_w[:, 2] < float(self.cfg.root_height_min_m))
            | (self._asset.data.root_link_pos_w[:, 2] > float(self.cfg.root_height_max_m))
            | (root_tilt_deg > float(self.cfg.root_tilt_threshold_deg))
            | (joint_margin < float(self.cfg.joint_limit_margin_rad))
            | (torque_ratio > float(self.cfg.torque_ratio_threshold))
        )
        self._safety_latched |= failure

    def _filtered_contact_active(self, matrix: torch.Tensor | None) -> torch.Tensor:
        if matrix is None or matrix.ndim != 4 or matrix.shape[0] != self.num_envs:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        finite = torch.isfinite(matrix).reshape(self.num_envs, -1).all(dim=-1)
        force_norms = torch.linalg.vector_norm(matrix, dim=-1).flatten(start_dim=1)
        if force_norms.shape[1] != len(self._forbidden_contact_sensor.cfg.filter_prim_paths_expr):
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        force_norms = force_norms.clone()
        force_norms[:, list(self._allowed_forbidden_filter_indices)] = 0.0
        return (force_norms.max(dim=-1).values > 1.0e-6) | ~finite

    def _resolve_env_ids(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None
    ) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)

    def _expand_rows(self, value: torch.Tensor, count: int, width: int, name: str) -> torch.Tensor:
        value = value.to(device=self.device, dtype=self._reference.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[0] == 1:
            value = value.expand(count, -1)
        if tuple(value.shape) != (count, width):
            raise ValueError(f"{name} shape mismatch: {tuple(value.shape)}")
        return value

    @staticmethod
    def _quat_yaw(quaternion: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quaternion.unbind(-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))

    def _compute_frame_pose(self, side: int) -> tuple[torch.Tensor, torch.Tensor]:
        return math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            self._asset.data.body_pos_w[:, self._body_ids[side]],
            self._asset.data.body_quat_w[:, self._body_ids[side]],
        )

    def _compute_frame_jacobian(self, side: int) -> torch.Tensor:
        joint_ids = self._left_jacobian_joint_ids if side == 0 else self._right_jacobian_joint_ids
        jacobian = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobian_body_ids[side], :, joint_ids
        ].clone()
        base_rotation = math_utils.matrix_from_quat(
            math_utils.quat_inv(self._asset.data.root_quat_w)
        )
        jacobian[:, :3] = torch.bmm(base_rotation, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(base_rotation, jacobian[:, 3:])
        return jacobian


class FrozenRecurrentLowerBodyAction(ActionTerm):
    """Zero-dimensional, non-trainable batched adapter for the certified student."""

    cfg: FrozenRecurrentLowerBodyActionCfg
    _asset: Articulation

    def __init__(self, cfg: FrozenRecurrentLowerBodyActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            list(cfg.joint_names), preserve_order=True
        )
        if len(self._joint_ids) != STUDENT_OUTPUT_DIM:
            raise RuntimeError(f"expected 12 lower-body joints, got {self._joint_names}")
        policy_path = retrieve_file_path(cfg.policy_path)
        actual_sha = sha256_file(policy_path)
        if actual_sha != cfg.expected_sha256:
            raise RuntimeError(f"certified checkpoint SHA mismatch: {actual_sha}")
        self.checkpoint_path = policy_path
        self.checkpoint_sha256 = actual_sha
        self._policy = torch.jit.load(policy_path, map_location=self.device).eval()
        for parameter in self._policy.parameters():
            parameter.requires_grad_(False)

        scale_cfg = {
            name: scale
            for name, scale in cfg.policy_output_scale.items()
            if any(token in name for token in ("hip", "knee", "ankle"))
        }
        indices, _, values = string_utils.resolve_matching_names_values(
            scale_cfg, self._joint_names
        )
        if set(indices) != set(range(STUDENT_OUTPUT_DIM)):
            raise RuntimeError("lower-body scale did not resolve all 12 joints")
        self._policy_output_scale = torch.ones(
            (self.num_envs, STUDENT_OUTPUT_DIM), device=self.device
        )
        self._policy_output_scale[:, indices] = torch.tensor(values, device=self.device)
        self._policy_output_offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = self._policy_output_offset.clone()
        self._policy_actions = torch.zeros((self.num_envs, STUDENT_OUTPUT_DIM), device=self.device)
        self._previous_policy_actions = torch.zeros_like(self._policy_actions)
        self._last_policy_input = torch.zeros(
            (self.num_envs, STUDENT_INPUT_DIM), device=self.device
        )
        self._hidden_state = torch.zeros(
            (1, self.num_envs, RECURRENT_STATE_DIM), device=self.device
        )
        self._cell_state = torch.zeros_like(self._hidden_state)
        self._fixed_command = torch.tensor(BASE_COMMAND, device=self.device).expand(
            self.num_envs, -1
        )

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def policy_actions(self) -> torch.Tensor:
        return self._policy_actions

    @property
    def previous_policy_actions(self) -> torch.Tensor:
        return self._previous_policy_actions

    @property
    def last_policy_input(self) -> torch.Tensor:
        return self._last_policy_input

    @property
    def hidden_state(self) -> torch.Tensor:
        return self._hidden_state

    @property
    def cell_state(self) -> torch.Tensor:
        return self._cell_state

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._raw_actions[env_ids] = 0.0
        self._policy_actions[env_ids] = 0.0
        self._previous_policy_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._policy_output_offset[env_ids]
        self._last_policy_input[env_ids] = 0.0
        self._hidden_state[:, env_ids] = 0.0
        self._cell_state[:, env_ids] = 0.0

    def restore_reference(self, env_ids: torch.Tensor, reference: dict[str, torch.Tensor]) -> None:
        count = len(env_ids)

        def expanded(name: str, shape: tuple[int, ...]) -> torch.Tensor:
            value = reference[name].to(self.device)
            if tuple(value.shape) == shape[1:]:
                value = value.unsqueeze(0)
            if value.shape[0] == 1:
                value = value.expand(count, *value.shape[1:])
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"lower reference {name} shape mismatch: {tuple(value.shape)} != {shape}"
                )
            return value

        self._hidden_state[:, env_ids] = expanded(
            "lower_hidden_state", (count, RECURRENT_STATE_DIM)
        ).unsqueeze(0)
        self._cell_state[:, env_ids] = expanded(
            "lower_cell_state", (count, RECURRENT_STATE_DIM)
        ).unsqueeze(0)
        self._previous_policy_actions[env_ids] = expanded(
            "previous_lower_policy_action", (count, STUDENT_OUTPUT_DIM)
        )
        self._policy_actions[env_ids] = self._previous_policy_actions[env_ids]
        self._processed_actions[env_ids] = (
            self._policy_actions[env_ids] * self._policy_output_scale[env_ids]
            + self._policy_output_offset[env_ids]
        )

    def capture_reference(self, env_id: int = 0) -> dict[str, torch.Tensor]:
        return {
            "lower_hidden_state": self._hidden_state[0, env_id].detach().cpu().clone(),
            "lower_cell_state": self._cell_state[0, env_id].detach().cpu().clone(),
            "previous_lower_policy_action": self._previous_policy_actions[env_id]
            .detach()
            .cpu()
            .clone(),
        }

    def process_actions(self, actions: torch.Tensor) -> None:
        if tuple(actions.shape) != (self.num_envs, 0):
            raise ValueError(f"lower action must be [N,0], got {tuple(actions.shape)}")
        student_observation = self._env.obs_buf[self.cfg.obs_group_name]
        if tuple(student_observation.shape) != (self.num_envs, STUDENT_OBSERVATION_DIM):
            raise RuntimeError(f"student observation mismatch: {tuple(student_observation.shape)}")
        self._last_policy_input.copy_(
            torch.cat(
                (self._fixed_command, student_observation, self._previous_policy_actions), dim=-1
            )
        )
        with torch.inference_mode():
            output, hidden, cell = batched_student_forward(
                self._policy, self._last_policy_input, self._hidden_state, self._cell_state
            )
        self._policy_actions.copy_(output)
        self._previous_policy_actions.copy_(output)
        self._hidden_state.copy_(hidden)
        self._cell_state.copy_(cell)
        self._processed_actions.copy_(
            output * self._policy_output_scale + self._policy_output_offset
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class HybridNormalApproachActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = HybridNormalApproachAction
    joint_names: Sequence[str] = ARM_JOINT_NAMES
    box_asset_name: str = "box"
    left_contact_sensor_name: str = "left_palm_box_contact"
    right_contact_sensor_name: str = "right_palm_box_contact"
    forbidden_contact_sensor_name: str = "robot_box_contact"
    control_dt_s: float = 0.02
    correction_scale_m: float = 0.005
    correction_rate_limit_m_per_control_step: float = 0.0001
    contacted_hand_additional_inward_cap_m: float = 0.001
    nominal_speed_limit_mps: float = 0.01
    nominal_acceleration_limit_mps2: float = 0.02
    nominal_jerk_limit_mps3: float = 0.10
    nominal_maximum_displacement_m: float = 0.09
    palm_orientation_object_wxyz: Sequence[float] = (
        0.7071067811865476,
        0.0,
        0.7071067811865476,
        0.0,
    )
    dls_damping_lambda: float = 0.01
    joint_target_rate_limit_rad_per_control_step: float = 0.005
    joint_limit_margin_rad: float = 0.10
    contact_force_threshold_n: float = 1.0
    force_peak_threshold_n: float = 9.81
    palm_impulse_threshold_ns: float = 0.981
    combined_impulse_threshold_ns: float = 1.962
    force_rate_threshold_nps: float = 490.5
    box_linear_speed_threshold_mps: float = 0.005
    box_angular_speed_threshold_radps: float = 0.008726646259971648
    box_translation_threshold_m: float = 0.005
    box_yaw_threshold_rad: float = 0.008726646259971648
    base_excursion_threshold_m: float = 0.05
    root_height_min_m: float = 0.5
    root_height_max_m: float = 1.0
    root_tilt_threshold_deg: float = 6.2075676918029785
    torque_ratio_threshold: float = 1.001
    impulse_window_steps: int = 5


# Stable imports for the coordinated runtime migration in this redesign commit.
ArmResidualAction = HybridNormalApproachAction
ArmResidualActionCfg = HybridNormalApproachActionCfg


@configclass
class FrozenRecurrentLowerBodyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FrozenRecurrentLowerBodyAction
    joint_names: Sequence[str] = ()
    obs_group_name: str = "student_policy"
    policy_path: str = ""
    expected_sha256: str = ""
    policy_output_scale: dict[str, float] = None  # type: ignore[assignment]
