"""Run the Stage-1 no-box G1 environment smoke test."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402

from g1_access_push.sim.stage1.no_box_env_cfg import (  # noqa: E402
    G1Stage1NoBoxEnvCfg,
)


def main() -> None:
    cfg = G1Stage1NoBoxEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device

    env = ManagerBasedEnv(cfg=cfg)
    env.reset(seed=args.seed)

    action_names = env.action_manager.active_terms
    action_dims = env.action_manager.action_term_dim

    print(f"action_terms={action_names}")
    print(f"action_dims={action_dims}")
    print(f"total_action_dim={env.action_manager.total_action_dim}")

    if "lower_body_joint_pos" not in action_names:
        raise RuntimeError("Missing lower_body_joint_pos action term")

    lower_index = action_names.index("lower_body_joint_pos")
    lower_start = sum(action_dims[:lower_index])
    lower_end = lower_start + action_dims[lower_index]

    if action_dims[lower_index] != 4:
        raise RuntimeError(
            f"Expected lower-body command dimension 4, got {action_dims[lower_index]}"
        )

    actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    # [vx, vy, wz, pelvis height]
    actions[:, lower_start:lower_end] = torch.tensor(
        [0.0, 0.0, 0.0, 0.72],
        device=env.device,
    )

    for step in range(args.num_steps):
        env.step(actions)

        robot = env.scene["robot"]

        finite = (
            torch.isfinite(robot.data.root_state_w).all()
            and torch.isfinite(robot.data.joint_pos).all()
            and torch.isfinite(robot.data.joint_vel).all()
        )

        if not bool(finite):
            raise RuntimeError(f"Non-finite state at step {step}")

    robot = env.scene["robot"]
    frames = env.scene["hand_frames"]

    root_position = robot.data.root_pos_w[0].detach().cpu()
    palm_positions = frames.data.target_pos_source[0].detach().cpu()

    print(f"root_position={root_position.tolist()}")
    print(f"palm_frame_names={frames.data.target_frame_names}")

    for name, position in zip(
        frames.data.target_frame_names,
        palm_positions,
        strict=True,
    ):
        print(f"{name}_in_pelvis={position.tolist()}")

    root_z = float(root_position[2])

    if not 0.50 < root_z < 1.00:
        raise RuntimeError(f"Invalid root height: {root_z}")

    print("STAGE1_S1_01_NO_BOX_SMOKE: PASS", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
