"""S2-03T arm-residual and frozen batched lower-body action terms.

The public policy action is exactly the 14 interleaved arm residuals.  The
certified locomotion student is evaluated internally and therefore contributes
zero dimensions to the policy action space.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
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
        raise ValueError(f"policy_input must be [N,{STUDENT_INPUT_DIM}], got {tuple(policy_input.shape)}")
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


class ArmResidualAction(ActionTerm):
    """Rate-limited joint residual around a frozen precontact IK target."""

    cfg: ArmResidualActionCfg
    _asset: Articulation

    def __init__(self, cfg: "ArmResidualActionCfg", env: "ManagerBasedEnv") -> None:
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

        body_ids, body_names = self._asset.find_bodies(
            ["left_hand_palm_link", "right_hand_palm_link"], preserve_order=True
        )
        if body_names != ["left_hand_palm_link", "right_hand_palm_link"]:
            raise RuntimeError(f"palm body order mismatch: {body_names}")
        self._body_ids = body_ids
        self._jacobian_body_ids = body_ids if not self._asset.is_fixed_base else [value - 1 for value in body_ids]
        self._left_jacobian_joint_ids = [value + 6 for value in self._left_joint_ids]
        self._right_jacobian_joint_ids = [value + 6 for value in self._right_joint_ids]

        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        self._left_ik = DifferentialIKController(ik_cfg, self.num_envs, self.device)
        self._right_ik = DifferentialIKController(ik_cfg, self.num_envs, self.device)

        shape = (self.num_envs, len(ARM_JOINT_NAMES))
        self._raw_actions = torch.zeros(shape, device=self.device)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._action_delta = torch.zeros_like(self._raw_actions)
        self._applied_residual = torch.zeros_like(self._raw_actions)
        self._reference = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._reference.clone()
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._target_lower = limits[..., 0] + float(cfg.joint_limit_margin_rad)
        self._target_upper = limits[..., 1] - float(cfg.joint_limit_margin_rad)
        if bool((self._target_lower >= self._target_upper).any()):
            raise RuntimeError("arm joint limits cannot preserve the frozen margin")

        self._bootstrap_mode = False
        self._bootstrap_commands = torch.zeros((self.num_envs, 2, 6), device=self.device)
        self._bootstrap_targets = torch.zeros_like(self._processed_actions)

    @property
    def action_dim(self) -> int:
        return len(ARM_JOINT_NAMES)

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
        return self._applied_residual

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
        if reference.shape == (1, self.action_dim):
            reference = reference.expand(self.num_envs, -1)
        if tuple(reference.shape) != tuple(self._reference.shape):
            raise ValueError(f"arm reference shape mismatch: {tuple(reference.shape)}")
        if not bool(torch.isfinite(reference).all()):
            raise ValueError("arm reference must be finite")
        if bool(((reference < self._target_lower) | (reference > self._target_upper)).any()):
            raise ValueError("arm reference violates the frozen 0.10 rad joint margin")
        self._reference.copy_(reference)
        self._applied_residual.zero_()
        self._processed_actions.copy_(reference)

    def restore_reference(self, env_ids: torch.Tensor, reference: torch.Tensor) -> None:
        """Restore a subset without perturbing recurrent episodes in other envs."""

        reference = reference.to(device=self.device, dtype=self._reference.dtype)
        if reference.ndim == 1:
            reference = reference.unsqueeze(0)
        if reference.shape[0] == 1:
            reference = reference.expand(len(env_ids), -1)
        if tuple(reference.shape) != (len(env_ids), self.action_dim):
            raise ValueError(f"arm subset reference shape mismatch: {tuple(reference.shape)}")
        self._reference[env_ids] = reference
        self._raw_actions[env_ids] = 0.0
        self._previous_raw_actions[env_ids] = 0.0
        self._action_delta[env_ids] = 0.0
        self._applied_residual[env_ids] = 0.0
        self._processed_actions[env_ids] = reference

    def set_bootstrap_mode(self, enabled: bool) -> None:
        self._bootstrap_mode = bool(enabled)

    def set_bootstrap_commands(self, commands: torch.Tensor) -> None:
        if tuple(commands.shape) != (self.num_envs, 2, 6):
            raise ValueError(f"bootstrap commands must be [N,2,6], got {tuple(commands.shape)}")
        self._bootstrap_commands.copy_(commands)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._previous_raw_actions[env_ids] = 0.0
        self._action_delta[env_ids] = 0.0
        self._applied_residual[env_ids] = 0.0
        self._processed_actions[env_ids] = self._reference[env_ids]
        self._bootstrap_commands[env_ids] = 0.0

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
        desired = clipped * float(self.cfg.residual_scale_rad)
        delta = torch.clamp(
            desired - self._applied_residual,
            -float(self.cfg.maximum_residual_change_rad),
            float(self.cfg.maximum_residual_change_rad),
        )
        self._applied_residual.add_(delta)
        target = self._reference + self._applied_residual
        self._processed_actions.copy_(torch.maximum(torch.minimum(target, self._target_upper), self._target_lower))

    def apply_actions(self) -> None:
        if not self._bootstrap_mode:
            self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
            return
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
        self._asset.set_joint_position_target(left_target, joint_ids=self._left_joint_ids)
        self._asset.set_joint_position_target(right_target, joint_ids=self._right_joint_ids)
        self._bootstrap_targets[:, 0::2] = left_target
        self._bootstrap_targets[:, 1::2] = right_target

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
        base_rotation = math_utils.matrix_from_quat(math_utils.quat_inv(self._asset.data.root_quat_w))
        jacobian[:, :3] = torch.bmm(base_rotation, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(base_rotation, jacobian[:, 3:])
        return jacobian


class FrozenRecurrentLowerBodyAction(ActionTerm):
    """Zero-dimensional, non-trainable batched adapter for the certified student."""

    cfg: FrozenRecurrentLowerBodyActionCfg
    _asset: Articulation

    def __init__(self, cfg: "FrozenRecurrentLowerBodyActionCfg", env: "ManagerBasedEnv") -> None:
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
        indices, _, values = string_utils.resolve_matching_names_values(scale_cfg, self._joint_names)
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
        self._last_policy_input = torch.zeros((self.num_envs, STUDENT_INPUT_DIM), device=self.device)
        self._hidden_state = torch.zeros(
            (1, self.num_envs, RECURRENT_STATE_DIM), device=self.device
        )
        self._cell_state = torch.zeros_like(self._hidden_state)
        self._fixed_command = torch.tensor(BASE_COMMAND, device=self.device).expand(self.num_envs, -1)

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
                raise ValueError(f"lower reference {name} shape mismatch: {tuple(value.shape)} != {shape}")
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
            "previous_lower_policy_action": self._previous_policy_actions[env_id].detach().cpu().clone(),
        }

    def process_actions(self, actions: torch.Tensor) -> None:
        if tuple(actions.shape) != (self.num_envs, 0):
            raise ValueError(f"lower action must be [N,0], got {tuple(actions.shape)}")
        student_observation = self._env.obs_buf[self.cfg.obs_group_name]
        if tuple(student_observation.shape) != (self.num_envs, STUDENT_OBSERVATION_DIM):
            raise RuntimeError(f"student observation mismatch: {tuple(student_observation.shape)}")
        self._last_policy_input.copy_(
            torch.cat((self._fixed_command, student_observation, self._previous_policy_actions), dim=-1)
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
class ArmResidualActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ArmResidualAction
    joint_names: Sequence[str] = ARM_JOINT_NAMES
    residual_scale_rad: float = 0.05
    maximum_residual_change_rad: float = 0.005
    joint_limit_margin_rad: float = 0.10


@configclass
class FrozenRecurrentLowerBodyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FrozenRecurrentLowerBodyAction
    joint_names: Sequence[str] = ()
    obs_group_name: str = "student_policy"
    policy_path: str = ""
    expected_sha256: str = ""
    policy_output_scale: dict[str, float] = None  # type: ignore[assignment]
