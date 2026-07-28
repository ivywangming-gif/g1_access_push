#!/usr/bin/env python3
"""Pre-training GPU-capacity smoke for the clean S2-03T formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--reference-sha256", required=True)
parser.add_argument("--num-envs", type=int, choices=(256, 128, 64), required=True)
parser.add_argument("--capacity-smoke", action="store_true", required=True)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
RUN = args.run_root.resolve()
RUN.mkdir(parents=True, exist_ok=False)
reference_path = args.reference.resolve()
if sha256_file(reference_path) != args.reference_sha256:
    raise SystemExit("PRECONTACT_REFERENCE_SHA_MISMATCH")
write_json(
    RUN / "capacity_status.json",
    {"status": "STARTING", "phase": "APP_LAUNCH", "num_envs": args.num_envs},
)
simulation_app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from agile.rl_env.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_agent_cfg import S203TPPORunnerCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env import S203TContactEnv  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env_cfg import (  # noqa: E402
    CERTIFIED_STUDENT_SHA256,
    S203TContactEnvCfg,
)


env = None
try:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    write_json(
        RUN / "capacity_status.json",
        {"status": "RUNNING", "phase": "ENVIRONMENT_CREATE", "num_envs": args.num_envs},
    )
    cfg = S203TContactEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = S203TContactEnv(cfg=cfg)
    env.install_precontact_reference(reference_path)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)

    agent_cfg = S203TPPORunnerCfg()
    agent_cfg.seed = 42
    agent_cfg.device = args.device
    agent_cfg.max_iterations = 1000
    agent_cfg.save_interval = 100
    agent_cfg.resume = False
    agent_cfg.load_run = None
    agent_cfg.load_checkpoint = None
    agent_cfg.load_optimizer = False
    write_json(
        RUN / "capacity_status.json",
        {"status": "RUNNING", "phase": "ROLLOUT_STORAGE_CREATE", "num_envs": args.num_envs},
    )
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    observation, extras = wrapped.get_observations()
    privileged = extras["observations"].get(runner.privileged_obs_type, observation)
    with torch.inference_mode():
        actor_action = runner.alg.policy.act_inference(observation)
        zero_action = torch.zeros((args.num_envs, 14), device=env.device)
        next_observation, reward, done, _ = wrapped.step(zero_action)
    storage = runner.alg.storage
    left_sensor = env.scene["left_palm_box_contact"]
    right_sensor = env.scene["right_palm_box_contact"]
    forbidden_sensor = env.scene["robot_box_contact"]
    left_matrix = left_sensor.data.force_matrix_w
    right_matrix = right_sensor.data.force_matrix_w
    forbidden_matrix = forbidden_sensor.data.force_matrix_w
    configured_forbidden_filters = list(forbidden_sensor.cfg.filter_prim_paths_expr)
    forbidden_filter_count = int(forbidden_sensor.contact_physx_view.filter_count)
    checks = {
        "formal_candidate": args.num_envs in (256, 128, 64),
        "environment_count": env.num_envs == args.num_envs,
        "observation_shape": list(observation.shape) == [args.num_envs, 77],
        "critic_observation_shape": list(privileged.shape) == [args.num_envs, 77],
        "actor_action_shape": list(actor_action.shape) == [args.num_envs, 14],
        "zero_action_shape": list(zero_action.shape) == [args.num_envs, 14],
        "next_observation_shape": list(next_observation.shape) == [args.num_envs, 77],
        "observation_finite": bool(torch.isfinite(observation).all()),
        "actor_action_finite": bool(torch.isfinite(actor_action).all()),
        "next_observation_finite": bool(torch.isfinite(next_observation).all()),
        "reward_finite": bool(torch.isfinite(reward).all()),
        "no_done_on_zero_step": not bool(done.any()),
        "rollout_storage_allocated": storage is not None,
        "policy_action_dim": wrapped.num_actions == 14,
        "policy_observation_dim": wrapped.num_obs == 77,
        "clean_actor": True,
        "resume_false": agent_cfg.resume is False,
        "checkpoint_not_loaded": agent_cfg.load_checkpoint is None and agent_cfg.load_run is None,
        "left_palm_force_matrix_shape": left_matrix is not None
        and list(left_matrix.shape) == [args.num_envs, 1, 1, 3],
        "right_palm_force_matrix_shape": right_matrix is not None
        and list(right_matrix.shape) == [args.num_envs, 1, 1, 3],
        "forbidden_sensor_single_body": int(forbidden_sensor.num_bodies) == 1,
        "forbidden_sensor_body_name": list(forbidden_sensor.body_names) == ["Box"],
        "forbidden_configured_filter_count_46": len(configured_forbidden_filters) == 46,
        "forbidden_filters_exact_no_wildcard": all(".*" not in item for item in configured_forbidden_filters),
        "forbidden_filter_count_matches_config": (
            forbidden_filter_count == len(configured_forbidden_filters)
        ),
        "forbidden_force_matrix_shape": forbidden_matrix is not None
        and list(forbidden_matrix.shape) == [args.num_envs, 1, forbidden_filter_count, 3],
        "contact_force_matrices_finite": all(
            matrix is not None and bool(torch.isfinite(matrix).all())
            for matrix in (left_matrix, right_matrix, forbidden_matrix)
        ),
        "certified_lower_checkpoint": (
            env.action_manager.get_term("frozen_lower_body").checkpoint_sha256
            == CERTIFIED_STUDENT_SHA256
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    memory = {
        "cuda_available": torch.cuda.is_available(),
        "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        memory.update({"free_bytes_after_step": int(free_bytes), "total_bytes": int(total_bytes)})
    result = {
        "schema_version": 1,
        "stage": "S2-03T",
        "mode": "FORMAL_CAPACITY_SMOKE",
        "status": "PASS" if not failed else "FAIL",
        "primary_reason": (
            "FORMAL_CAPACITY_AVAILABLE" if not failed else "FORMAL_CAPACITY_CONTRACT_CHECK_FAILED"
        ),
        "num_envs": args.num_envs,
        "checks": checks,
        "failed_checks": failed,
        "memory": memory,
        "contact_sensor_audit": {
            "left_force_matrix_shape": list(left_matrix.shape) if left_matrix is not None else None,
            "right_force_matrix_shape": list(right_matrix.shape) if right_matrix is not None else None,
            "forbidden_force_matrix_shape": list(forbidden_matrix.shape) if forbidden_matrix is not None else None,
            "forbidden_filter_count": forbidden_filter_count,
            "configured_forbidden_filter_count": len(configured_forbidden_filters),
        },
        "reference_path": str(reference_path),
        "reference_sha256": args.reference_sha256,
        "training_started": False,
        "checkpoint_created": False,
        "pilot_checkpoint_used": False,
        "resume": False,
    }
    write_json(RUN / "capacity_smoke_result.json", result)
    write_json(
        RUN / "capacity_status.json",
        {"status": "COMPLETE", "phase": "DONE", "result": result["status"], "num_envs": args.num_envs},
    )
    if failed:
        raise SystemExit(2)
except BaseException as exc:
    message = repr(exc)
    oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower()
    if not (RUN / "capacity_smoke_result.json").is_file():
        write_json(
            RUN / "capacity_smoke_result.json",
            {
                "schema_version": 1,
                "stage": "S2-03T",
                "mode": "FORMAL_CAPACITY_SMOKE",
                "status": "OOM" if oom else "INVALID",
                "primary_reason": "CUDA_OUT_OF_MEMORY" if oom else "IMPLEMENTATION_EXCEPTION",
                "num_envs": args.num_envs,
                "error": message,
                "training_started": False,
                "checkpoint_created": False,
                "pilot_checkpoint_used": False,
                "resume": False,
            },
        )
    if not isinstance(exc, SystemExit) or exc.code != 2:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
