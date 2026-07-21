"""Smoke-test the recurrent student inside the complete no-box G1."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_json",
    type=Path,
    required=True,
)
parser.add_argument(
    "--steps",
    type=int,
    default=200,
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402

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


def root_tilt_deg(quaternion: torch.Tensor) -> torch.Tensor:
    x = quaternion[..., 1]
    y = quaternion[..., 2]

    cosine = 1.0 - 2.0 * (
        x.square() + y.square()
    )
    cosine = torch.clamp(cosine, -1.0, 1.0)

    return torch.rad2deg(torch.acos(cosine))


def main() -> None:
    cfg = G1Stage1NoBoxRecurrentEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.sim.device = args.device

    env = ManagerBasedEnv(cfg=cfg)

    try:
        env.reset(seed=cfg.seed)

        robot = env.scene["robot"]

        names = env.action_manager.active_terms
        dims = env.action_manager.action_term_dim

        upper_slice = term_slice(
            names,
            dims,
            "upper_body_joint_pos",
        )
        lower_slice = term_slice(
            names,
            dims,
            "lower_body_joint_pos",
        )

        lower_term = env.action_manager.get_term(
            "lower_body_joint_pos"
        )

        actions = torch.zeros(
            (1, env.action_manager.total_action_dim),
            device=env.device,
        )
        actions[:, lower_slice] = torch.tensor(
            [0.0, 0.0, 0.0, 0.72],
            dtype=torch.float32,
            device=env.device,
        )
        actions[:, upper_slice] = 0.0

        root_heights: list[torch.Tensor] = []
        root_tilts: list[torch.Tensor] = []

        all_finite = True

        for step in range(args.steps):
            env.step(actions)

            tensors = (
                robot.data.root_state_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
                robot.data.applied_torque,
                lower_term.policy_actions,
                lower_term.processed_actions,
                lower_term.last_policy_input,
            )

            if not all(
                bool(torch.isfinite(value).all())
                for value in tensors
            ):
                all_finite = False
                raise RuntimeError(
                    f"Non-finite state at step {step}"
                )

            root_heights.append(
                robot.data.root_pos_w[0, 2].clone()
            )
            root_tilts.append(
                root_tilt_deg(
                    robot.data.root_quat_w[0].unsqueeze(0)
                )[0]
            )

        root_heights_t = torch.stack(root_heights)
        root_tilts_t = torch.stack(root_tilts)

        metrics = {
            "steps": args.steps,
            "duration_s": args.steps * env.step_dt,
            "step_dt_s": env.step_dt,
            "action_terms": names,
            "action_dims": dims,
            "total_action_dim": (
                env.action_manager.total_action_dim
            ),
            "policy_input_shape": list(
                lower_term.last_policy_input.shape
            ),
            "policy_output_shape": list(
                lower_term.policy_actions.shape
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
            "hidden_state_max": (
                lower_term.recurrent_state_max
            ),
            "cell_state_max": (
                lower_term.recurrent_cell_max
            ),
            "all_numeric_finite": all_finite,
        }

        checks = {
            "finite": all_finite,
            "action_terms": names
            == [
                "upper_body_joint_pos",
                "lower_body_joint_pos",
            ],
            "action_dims": dims == [17, 4],
            "policy_input": metrics[
                "policy_input_shape"
            ]
            == [80],
            "policy_output": metrics[
                "policy_output_shape"
            ]
            == [1, 12],
            "root_height": (
                metrics["root_height_min_m"] > 0.50
                and metrics["root_height_max_m"] < 1.00
            ),
            "root_tilt": (
                metrics["root_tilt_max_deg"] <= 15.0
            ),
            "hidden_state_changed": (
                metrics["hidden_state_max"] > 0.0
            ),
            "cell_state_changed": (
                metrics["cell_state_max"] > 0.0
            ),
        }

        passed = all(checks.values())

        result = {
            "passed": passed,
            "checks": checks,
            "metrics": metrics,
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
            + "\n"
        )

        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )
        print(
            "RECURRENT_STUDENT_ADAPTER_SMOKE: "
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
