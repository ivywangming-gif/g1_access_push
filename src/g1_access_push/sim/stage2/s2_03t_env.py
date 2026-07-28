"""S2-03T ManagerBasedRLEnv with exact precontact reset restoration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from isaaclab.envs import ManagerBasedRLEnv

from g1_access_push.sim.stage2.s2_03t_actions import (
    ARM_JOINT_NAMES,
    ArmResidualAction,
    FrozenRecurrentLowerBodyAction,
)
from g1_access_push.sim.stage2.s2_03t_mdp import runtime_state


REFERENCE_SCHEMA_VERSION = 1


class S203TContactEnv(ManagerBasedRLEnv):
    """Adds reference installation and reset-before-observation terminal caching."""

    def __init__(self, cfg, render_mode: str | None = None, **_: Any) -> None:
        self._s2_03t_reference: dict[str, Any] | None = None
        self._s2_03t_terminal_snapshots: dict[int, dict[str, Any]] = {}
        self._s2_03t_bootstrap_mode = False
        super().__init__(cfg)
        self.render_mode = render_mode
        self._configure_box_mass_properties()

    @property
    def precontact_reference(self) -> dict[str, Any] | None:
        return self._s2_03t_reference

    def installed_reference_tensor_diffs(self, env_id: int = 0) -> dict[str, float]:
        """Compare the simulator state after reset with the installed frozen reference."""

        if self._s2_03t_reference is None:
            raise RuntimeError("PRECONTACT_REFERENCE_NOT_INSTALLED")
        if not 0 <= env_id < self.num_envs:
            raise IndexError(f"environment index out of range: {env_id}")
        reference = self._s2_03t_reference
        robot = self.scene["robot"]
        box = self.scene["box"]
        arm: ArmResidualAction = self.action_manager.get_term("arm_residual")
        lower: FrozenRecurrentLowerBodyAction = self.action_manager.get_term("frozen_lower_body")
        robot_root = robot.data.root_state_w[env_id].detach().clone()
        robot_root[:3] -= self.scene.env_origins[env_id]
        box_root = box.data.root_state_w[env_id].detach().clone()
        box_root[:3] -= self.scene.env_origins[env_id]
        actual = {
            "robot_root_state_relative": robot_root,
            "robot_joint_position": robot.data.joint_pos[env_id],
            "robot_joint_velocity": robot.data.joint_vel[env_id],
            "box_root_state_relative": box_root,
            "arm_ik_target": arm.reference[env_id],
            "lower_hidden_state": lower.hidden_state[0, env_id],
            "lower_cell_state": lower.cell_state[0, env_id],
            "previous_lower_policy_action": lower.previous_policy_actions[env_id],
        }
        return {
            name: float(torch.max(torch.abs(value.detach().cpu() - reference[name])))
            for name, value in actual.items()
        }

    def install_precontact_reference(self, source: Path | dict[str, Any]) -> None:
        if isinstance(source, Path):
            reference = torch.load(source, map_location="cpu", weights_only=True)
        else:
            reference = source
        self._validate_reference(reference)
        self._s2_03t_reference = reference
        arm: ArmResidualAction = self.action_manager.get_term("arm_residual")
        arm.set_reference(reference["arm_ik_target"].to(self.device))
        self.reset(seed=42)

    def pop_terminal_snapshot(self, env_id: int) -> dict[str, Any] | None:
        return self._s2_03t_terminal_snapshots.pop(int(env_id), None)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        state = getattr(self, "_s2_03t_runtime_state", None)
        if state is not None and hasattr(self, "episode_length_buf"):
            completed = ids[self.episode_length_buf[ids] > 0]
            if len(completed) > 0 and not self._s2_03t_bootstrap_mode:
                self._s2_03t_terminal_snapshots.update(state.snapshot(completed))
        super()._reset_idx(ids)
        if self._s2_03t_reference is not None:
            self._restore_reference(ids)

    def _restore_reference(self, env_ids: torch.Tensor) -> None:
        assert self._s2_03t_reference is not None
        reference = self._s2_03t_reference
        robot = self.scene["robot"]
        box = self.scene["box"]
        count = len(env_ids)

        robot_root = reference["robot_root_state_relative"].to(self.device).reshape(1, 13).expand(count, -1).clone()
        robot_root[:, :3] += self.scene.env_origins[env_ids]
        robot.write_root_state_to_sim(robot_root, env_ids=env_ids)
        joint_pos = reference["robot_joint_position"].to(self.device).reshape(1, -1).expand(count, -1)
        joint_vel = reference["robot_joint_velocity"].to(self.device).reshape(1, -1).expand(count, -1)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        box_root = reference["box_root_state_relative"].to(self.device).reshape(1, 13).expand(count, -1).clone()
        box_root[:, :3] += self.scene.env_origins[env_ids]
        box.write_root_state_to_sim(box_root, env_ids=env_ids)

        arm: ArmResidualAction = self.action_manager.get_term("arm_residual")
        lower: FrozenRecurrentLowerBodyAction = self.action_manager.get_term("frozen_lower_body")
        arm.restore_reference(env_ids, reference["arm_ik_target"])
        lower.restore_reference(env_ids, reference)
        runtime_state(self).reset(env_ids)

    def _configure_box_mass_properties(self) -> None:
        box = self.scene["box"]
        view = getattr(box, "root_physx_view", None) or getattr(box, "root_view", None)
        if view is None:
            raise RuntimeError("RUNTIME_MASS_VIEW_UNAVAILABLE")
        masses = view.get_masses().clone()
        coms = view.get_coms().clone()
        inertias = view.get_inertias().clone()
        masses[...] = 5.0
        coms[..., :3] = torch.tensor((0.0, 0.0, -0.4), device=coms.device, dtype=coms.dtype)
        coms[..., 3:7] = torch.tensor((0.0, 0.0, 0.0, 1.0), device=coms.device, dtype=coms.dtype)
        inertias.zero_()
        for index, value in zip((0, 4, 8), (0.12083333333333333, 0.43333333333333335, 0.5208333333333334), strict=True):
            inertias[..., index] = value
        indices = torch.arange(int(view.count), dtype=torch.int32, device="cpu")
        view.set_masses(masses, indices)
        view.set_inertias(inertias, indices)
        view.set_coms(coms, indices)
        actual_mass = view.get_masses()
        actual_com = view.get_coms()
        actual_inertia = view.get_inertias()
        checks = (
            torch.allclose(actual_mass, masses, atol=1.0e-5, rtol=0.0),
            torch.allclose(actual_com, coms, atol=1.0e-5, rtol=0.0),
            torch.allclose(actual_inertia, inertias, atol=1.0e-5, rtol=0.0),
        )
        if not all(checks):
            raise RuntimeError("RUNTIME_MASS_PROPERTIES_FAILED")
        self._s2_03t_mass_audit = {
            "mass_kg": 5.0,
            "com_local_xyz_m": [0.0, 0.0, -0.4],
            "diagonal_inertia_kg_m2": [0.12083333333333333, 0.43333333333333335, 0.5208333333333334],
            "instance_count": int(view.count),
            "status": "PASS",
        }

    def _validate_reference(self, reference: dict[str, Any]) -> None:
        required = {
            "schema_version",
            "arm_joint_names",
            "robot_joint_names",
            "robot_root_state_relative",
            "robot_joint_position",
            "robot_joint_velocity",
            "box_root_state_relative",
            "arm_ik_target",
            "lower_hidden_state",
            "lower_cell_state",
            "previous_lower_policy_action",
            "source",
        }
        missing = sorted(required - set(reference))
        if missing:
            raise ValueError(f"precontact reference missing: {missing}")
        if reference["schema_version"] != REFERENCE_SCHEMA_VERSION:
            raise ValueError("precontact reference schema mismatch")
        if tuple(reference["arm_joint_names"]) != ARM_JOINT_NAMES:
            raise ValueError("precontact arm order mismatch")
        if list(reference["robot_joint_names"]) != list(self.scene["robot"].joint_names):
            raise ValueError("precontact full joint order mismatch")
        shapes = {
            "robot_root_state_relative": (13,),
            "robot_joint_position": (self.scene["robot"].num_joints,),
            "robot_joint_velocity": (self.scene["robot"].num_joints,),
            "box_root_state_relative": (13,),
            "arm_ik_target": (14,),
            "lower_hidden_state": (256,),
            "lower_cell_state": (256,),
            "previous_lower_policy_action": (12,),
        }
        for name, shape in shapes.items():
            value = reference[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"precontact {name} shape mismatch")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"precontact {name} is non-finite")
