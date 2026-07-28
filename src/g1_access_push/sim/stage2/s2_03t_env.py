"""S2-03T ManagerBasedRLEnv with exact precontact reset restoration."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv

from g1_access_push.sim.stage2.s2_03t_actions import (
    ARM_JOINT_NAMES,
    ArmResidualAction,
    FrozenRecurrentLowerBodyAction,
)
from g1_access_push.sim.stage2.s2_03t_mdp import runtime_state

REFERENCE_SCHEMA_VERSION = 1
FULL_GAP_ENDPOINT_PROBABILITY = 0.05


class S203TContactEnv(ManagerBasedRLEnv):
    """Adds reference installation and reset-before-observation terminal caching."""

    def __init__(self, cfg, render_mode: str | None = None, **_: Any) -> None:
        self._s2_03t_reference: dict[str, Any] | None = None
        self._s2_03t_terminal_snapshots: dict[int, dict[str, Any]] = {}
        self._s2_03t_completed_episodes: deque[dict[str, Any]] = deque()
        self._s2_03t_pending_curriculum_episodes: list[dict[str, Any]] = []
        self._s2_03t_curriculum_window: deque[dict[str, Any]] = deque(maxlen=2048)
        self._s2_03t_curriculum_promotions: list[dict[str, Any]] = []
        self._s2_03t_curriculum_enabled = False
        self._s2_03t_curriculum_level = 0
        self._s2_03t_curriculum_maximum_level = 0
        self._s2_03t_fixed_initial_gap_m = 0.06
        self._s2_03t_sampled_initial_gap_m: torch.Tensor | None = None
        self._s2_03t_bootstrap_mode = False
        super().__init__(cfg)
        self.render_mode = render_mode
        self._s2_03t_sampled_initial_gap_m = torch.full(
            (self.num_envs,), 0.06, dtype=torch.float32, device=self.device
        )
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

    def drain_completed_episodes(self) -> list[dict[str, Any]]:
        completed = list(self._s2_03t_completed_episodes)
        self._s2_03t_completed_episodes.clear()
        return completed

    @property
    def curriculum_level(self) -> int:
        return int(self._s2_03t_curriculum_level)

    @property
    def curriculum_promotions(self) -> list[dict[str, Any]]:
        return list(self._s2_03t_curriculum_promotions)

    @property
    def sampled_initial_gap_m(self) -> torch.Tensor:
        if self._s2_03t_sampled_initial_gap_m is None:
            self._s2_03t_sampled_initial_gap_m = torch.full(
                (self.num_envs,), 0.06, dtype=torch.float32, device=self.device
            )
        return self._s2_03t_sampled_initial_gap_m

    def configure_contact_curriculum(
        self,
        *,
        enabled: bool,
        maximum_level: int,
        fixed_gap_m: float = 0.06,
    ) -> None:
        if maximum_level not in (0, 1, 2, 3):
            raise ValueError(f"curriculum maximum level must be 0..3, got {maximum_level}")
        if not 0.0 <= fixed_gap_m <= 0.06:
            raise ValueError(f"fixed initial gap must be in [0,0.06], got {fixed_gap_m}")
        self._s2_03t_curriculum_enabled = bool(enabled)
        self._s2_03t_curriculum_maximum_level = int(maximum_level)
        self._s2_03t_fixed_initial_gap_m = float(fixed_gap_m)
        self._s2_03t_curriculum_level = min(
            self._s2_03t_curriculum_level,
            self._s2_03t_curriculum_maximum_level,
        )
        self._s2_03t_curriculum_window.clear()

    def update_contact_curriculum(self, env_ids: Sequence[int] | torch.Tensor) -> dict[str, float]:
        """Consume completed episodes, promote sequentially, and sample reset gaps."""

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for snapshot in self._s2_03t_pending_curriculum_episodes:
            reasons = set(snapshot.get("termination_reasons", []))
            pushing = bool(
                reasons
                & {
                    "box_linear_speed",
                    "box_angular_speed",
                    "box_translation",
                    "box_yaw_change",
                }
            )
            hard_safety = bool(
                reasons
                & {
                    "forbidden_non_palm_box_collision",
                    "force_peak",
                    "palm_impulse",
                    "combined_impulse",
                    "force_rate",
                    "base_excursion",
                    "root_height",
                    "root_tilt",
                    "arm_joint_margin",
                    "arm_torque",
                }
            )
            self._s2_03t_curriculum_window.append(
                {
                    "success": bool(snapshot.get("success", False)),
                    "hard_safety": hard_safety,
                    "pushing": pushing,
                    "nonfinite": not bool(snapshot.get("finite", False)),
                }
            )
        self._s2_03t_pending_curriculum_episodes.clear()

        if (
            self._s2_03t_curriculum_enabled
            and self._s2_03t_curriculum_level < self._s2_03t_curriculum_maximum_level
            and len(self._s2_03t_curriculum_window) == 2048
        ):
            window = list(self._s2_03t_curriculum_window)
            count = float(len(window))
            success_fraction = sum(bool(item["success"]) for item in window) / count
            hard_fraction = sum(bool(item["hard_safety"]) for item in window) / count
            push_fraction = sum(bool(item["pushing"]) for item in window) / count
            nonfinite_count = sum(bool(item["nonfinite"]) for item in window)
            threshold = (0.80, 0.75, 0.65)[self._s2_03t_curriculum_level]
            if (
                success_fraction >= threshold
                and hard_fraction <= 0.005
                and push_fraction <= 0.001
                and nonfinite_count == 0
            ):
                source_level = self._s2_03t_curriculum_level
                self._s2_03t_curriculum_level += 1
                self._s2_03t_curriculum_promotions.append(
                    {
                        "source_level": source_level,
                        "target_level": self._s2_03t_curriculum_level,
                        "common_step_counter": int(self.common_step_counter),
                        "success_fraction": success_fraction,
                        "hard_safety_fraction": hard_fraction,
                        "pushing_fraction": push_fraction,
                        "nonfinite_count": nonfinite_count,
                    }
                )
                self._s2_03t_curriculum_window.clear()

        if self._s2_03t_curriculum_enabled:
            bounds = ((0.0, 0.005), (0.0, 0.010), (0.010, 0.030), (0.030, 0.060))
            minimum, maximum = bounds[self._s2_03t_curriculum_level]
            samples = minimum + (maximum - minimum) * torch.rand(
                len(ids), dtype=torch.float32, device=self.device
            )
            if self._s2_03t_curriculum_level == 3:
                endpoint_mask = (
                    torch.rand(len(ids), device=self.device) < FULL_GAP_ENDPOINT_PROBABILITY
                )
                samples = torch.where(endpoint_mask, torch.full_like(samples, 0.06), samples)
        else:
            samples = torch.full(
                (len(ids),),
                self._s2_03t_fixed_initial_gap_m,
                dtype=torch.float32,
                device=self.device,
            )
        self.sampled_initial_gap_m[ids] = samples
        return {
            "level": float(self._s2_03t_curriculum_level),
            "completed_window": float(len(self._s2_03t_curriculum_window)),
            "mean_sampled_gap_m": float(samples.mean()) if len(ids) else 0.0,
        }

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        state = getattr(self, "_s2_03t_runtime_state", None)
        if state is not None and hasattr(self, "episode_length_buf"):
            completed = ids[self.episode_length_buf[ids] > 0]
            if len(completed) > 0 and not self._s2_03t_bootstrap_mode:
                snapshots = state.snapshot(completed)
                self._s2_03t_terminal_snapshots.update(snapshots)
                ordered = [snapshots[int(env_id)] for env_id in completed.tolist()]
                self._s2_03t_completed_episodes.extend(ordered)
                self._s2_03t_pending_curriculum_episodes.extend(ordered)
        super()._reset_idx(ids)
        if self._s2_03t_reference is not None:
            self._restore_reference(ids)

    def _restore_reference(self, env_ids: torch.Tensor) -> None:
        assert self._s2_03t_reference is not None
        reference = self._s2_03t_reference
        robot = self.scene["robot"]
        box = self.scene["box"]
        count = len(env_ids)

        robot_root = (
            reference["robot_root_state_relative"]
            .to(self.device)
            .reshape(1, 13)
            .expand(count, -1)
            .clone()
        )
        robot_root[:, :3] += self.scene.env_origins[env_ids]
        robot.write_root_state_to_sim(robot_root, env_ids=env_ids)
        joint_pos = (
            reference["robot_joint_position"].to(self.device).reshape(1, -1).expand(count, -1)
        )
        joint_vel = (
            reference["robot_joint_velocity"].to(self.device).reshape(1, -1).expand(count, -1)
        )
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        box_root = (
            reference["box_root_state_relative"]
            .to(self.device)
            .reshape(1, 13)
            .expand(count, -1)
            .clone()
        )
        box_root[:, :3] += self.scene.env_origins[env_ids]
        object_x_world = math_utils.quat_apply(
            box_root[:, 3:7],
            torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(count, -1),
        )
        shift = (0.06 - self.sampled_initial_gap_m[env_ids]).unsqueeze(-1) * object_x_world
        box_root[:, :3] -= shift
        box.write_root_state_to_sim(box_root, env_ids=env_ids)

        arm: ArmResidualAction = self.action_manager.get_term("arm_residual")
        lower: FrozenRecurrentLowerBodyAction = self.action_manager.get_term("frozen_lower_body")
        arm.restore_reference(env_ids, reference["arm_ik_target"])
        lower.restore_reference(env_ids, reference)
        arm.set_episode_anchor(env_ids, box_root[:, :3], box_root[:, 3:7])
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
        for index, value in zip(
            (0, 4, 8), (0.12083333333333333, 0.43333333333333335, 0.5208333333333334), strict=True
        ):
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
            "diagonal_inertia_kg_m2": [
                0.12083333333333333,
                0.43333333333333335,
                0.5208333333333334,
            ],
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
