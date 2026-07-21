"""Single-environment adapter for the exported G1 recurrent locomotion student."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import retrieve_file_path

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


BASE_COMMAND_DIM = 4
STUDENT_OBSERVATION_DIM = 64
PREVIOUS_POLICY_ACTION_DIM = 12
STUDENT_INPUT_DIM = 80
STUDENT_OUTPUT_DIM = 12


def compose_recurrent_student_input(
    base_command: torch.Tensor,
    student_observation: torch.Tensor,
    previous_policy_action: torch.Tensor,
) -> torch.Tensor:
    """Compose the one-dimensional 80-value input required by the LSTM."""

    expected_shapes = {
        "base_command": (1, BASE_COMMAND_DIM),
        "student_observation": (1, STUDENT_OBSERVATION_DIM),
        "previous_policy_action": (1, PREVIOUS_POLICY_ACTION_DIM),
    }
    actual_shapes = {
        "base_command": tuple(base_command.shape),
        "student_observation": tuple(student_observation.shape),
        "previous_policy_action": tuple(previous_policy_action.shape),
    }

    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {actual_shapes[name]}. "
                "The exported recurrent student supports exactly one environment."
            )

    policy_input = torch.cat(
        (
            base_command[0],
            student_observation[0],
            previous_policy_action[0],
        ),
        dim=0,
    )

    if tuple(policy_input.shape) != (STUDENT_INPUT_DIM,):
        raise RuntimeError(
            f"Policy input must have shape ({STUDENT_INPUT_DIM},), "
            f"got {tuple(policy_input.shape)}."
        )

    return policy_input


class RecurrentStudentLowerBodyAction(ActionTerm):
    """Convert [vx, vy, wz, height] into 12 G1 leg joint targets."""

    cfg: RecurrentStudentLowerBodyActionCfg
    _asset: Articulation

    def __init__(
        self,
        cfg: RecurrentStudentLowerBodyActionCfg,
        env: ManagerBasedEnv,
    ) -> None:
        super().__init__(cfg, env)

        if self.num_envs != 1:
            raise ValueError(
                "The exported recurrent student stores one LSTM state internally "
                f"and requires num_envs=1, got {self.num_envs}."
            )

        if not cfg.joint_names:
            raise ValueError("joint_names must not be empty.")

        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names
        )

        if len(self._joint_ids) != STUDENT_OUTPUT_DIM:
            raise ValueError(
                f"Expected {STUDENT_OUTPUT_DIM} leg joints, "
                f"resolved {len(self._joint_ids)}: {self._joint_names}"
            )

        policy_path = retrieve_file_path(cfg.policy_path)
        self._policy = torch.jit.load(
            policy_path,
            map_location=self.device,
        )
        self._policy.eval()
        self._policy.to(self.device)

        for state_name in ("hidden_state", "cell_state"):
            state = getattr(self._policy, state_name, None)

            if not isinstance(state, torch.Tensor):
                raise RuntimeError(
                    f"Recurrent model is missing buffer {state_name!r}."
                )

            if tuple(state.shape) != (1, 256):
                raise RuntimeError(
                    f"{state_name} must have shape (1, 256), "
                    f"got {tuple(state.shape)}."
                )

        command_names = ("vx", "vy", "wz", "height")
        missing_limits = [
            name
            for name in command_names
            if name not in cfg.command_limits
        ]

        if missing_limits:
            raise ValueError(
                f"Missing command limits: {missing_limits}"
            )

        self._command_min = torch.tensor(
            [[cfg.command_limits[name][0] for name in command_names]],
            dtype=torch.float32,
            device=self.device,
        )
        self._command_max = torch.tensor(
            [[cfg.command_limits[name][1] for name in command_names]],
            dtype=torch.float32,
            device=self.device,
        )

        scale_cfg = {
            name: scale
            for name, scale in cfg.policy_output_scale.items()
            if any(
                token in name
                for token in ("hip", "knee", "ankle")
            )
        }

        index_list, _, value_list = (
            string_utils.resolve_matching_names_values(
                scale_cfg,
                self._joint_names,
            )
        )

        resolved_indices = set(index_list)

        if resolved_indices != set(range(STUDENT_OUTPUT_DIM)):
            unresolved = [
                name
                for index, name in enumerate(self._joint_names)
                if index not in resolved_indices
            ]
            raise ValueError(
                "policy_output_scale did not resolve every leg joint. "
                f"Unresolved: {unresolved}"
            )

        self._policy_output_scale = torch.ones(
            (1, STUDENT_OUTPUT_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self._policy_output_scale[:, index_list] = torch.tensor(
            value_list,
            dtype=torch.float32,
            device=self.device,
        )

        self._policy_output_offset = (
            self._asset.data.default_joint_pos[
                :, self._joint_ids
            ].clone()
        )

        self._raw_actions = torch.zeros(
            (1, BASE_COMMAND_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self._policy_actions = torch.zeros(
            (1, STUDENT_OUTPUT_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self._previous_policy_actions = torch.zeros_like(
            self._policy_actions
        )
        self._processed_actions = torch.zeros_like(
            self._policy_actions
        )
        self._last_policy_input = torch.zeros(
            STUDENT_INPUT_DIM,
            dtype=torch.float32,
            device=self.device,
        )

        self.reset()

    @property
    def action_dim(self) -> int:
        return BASE_COMMAND_DIM

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
    def last_policy_input(self) -> torch.Tensor:
        return self._last_policy_input

    @property
    def recurrent_state_max(self) -> float:
        return float(
            self._policy.hidden_state.detach().abs().max().item()
        )

    @property
    def recurrent_cell_max(self) -> float:
        return float(
            self._policy.cell_state.detach().abs().max().item()
        )

    def _zero_recurrent_state(self) -> None:
        with torch.no_grad():
            self._policy.hidden_state.zero_()
            self._policy.cell_state.zero_()

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        # This adapter explicitly supports one environment only.
        del env_ids

        self._zero_recurrent_state()
        self._raw_actions.zero_()
        self._policy_actions.zero_()
        self._previous_policy_actions.zero_()
        self._processed_actions.zero_()
        self._last_policy_input.zero_()

    def process_actions(self, actions: torch.Tensor) -> None:
        if tuple(actions.shape) != (1, BASE_COMMAND_DIM):
            raise ValueError(
                f"Expected command shape (1, {BASE_COMMAND_DIM}), "
                f"got {tuple(actions.shape)}."
            )

        self._raw_actions.copy_(actions)

        base_command = torch.clamp(
            actions,
            min=self._command_min,
            max=self._command_max,
        )

        student_observation = self._env.obs_buf[
            self.cfg.obs_group_name
        ]

        policy_input = compose_recurrent_student_input(
            base_command,
            student_observation,
            self._previous_policy_actions,
        )
        self._last_policy_input.copy_(policy_input)

        with torch.inference_mode():
            policy_output = self._policy(policy_input)

        if tuple(policy_output.shape) != (STUDENT_OUTPUT_DIM,):
            raise RuntimeError(
                f"Expected output shape ({STUDENT_OUTPUT_DIM},), "
                f"got {tuple(policy_output.shape)}."
            )

        if not bool(torch.isfinite(policy_output).all()):
            raise RuntimeError(
                "Recurrent student produced non-finite output."
            )

        self._policy_actions.copy_(
            policy_output.unsqueeze(0)
        )
        self._processed_actions.copy_(
            self._policy_actions * self._policy_output_scale
            + self._policy_output_offset
        )

        # This is the 12-dimensional previous-action observation
        # required at the next control step.
        self._previous_policy_actions.copy_(
            self._policy_actions
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(
            self._processed_actions,
            joint_ids=self._joint_ids,
        )


@configclass
class RecurrentStudentLowerBodyActionCfg(ActionTermCfg):
    """Configuration for RecurrentStudentLowerBodyAction."""

    class_type: type[ActionTerm] = (
        RecurrentStudentLowerBodyAction
    )

    joint_names: list[str] = field(default_factory=list)
    obs_group_name: str = "student_policy"
    policy_path: str = ""
    policy_output_scale: dict[str, float] = field(
        default_factory=dict
    )
    command_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "vx": (-0.20, 0.50),
            "vy": (-0.40, 0.40),
            "wz": (-0.50, 0.50),
            "height": (0.65, 0.72),
        }
    )
