"""Stage 2 IsaacLab task registrations."""

import gymnasium as gym


TASK_ID = "G1-S2-03T-Contact-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="g1_access_push.sim.stage2.s2_03t_env:S203TContactEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "g1_access_push.sim.stage2.s2_03t_env_cfg:S203TContactEnvCfg",
            "rsl_rl_cfg_entry_point": "g1_access_push.sim.stage2.s2_03t_agent_cfg:S203TPPORunnerCfg",
        },
    )


__all__ = ["TASK_ID"]
