#!/usr/bin/env python3
"""Clean, no-resume S2-03T pilot or formal PPO training process."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
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
parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--max-iterations", type=int, required=True)
parser.add_argument("--save-interval", type=int, required=True)
parser.add_argument("--seed", type=int, default=42)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seed != 42:
    parser.error("S2-03T clean actor seed is frozen at 42")
if args.mode == "pilot" and args.num_envs != 64:
    parser.error("pilot requires exactly 64 environments")
if args.mode == "pilot" and args.max_iterations not in (25, 50, 75, 100):
    parser.error("pilot target must be one of 25/50/75/100 clean iterations")
if args.mode == "formal" and args.num_envs not in (256, 128, 64):
    parser.error("formal num_envs must be one frozen capacity candidate")
if args.mode == "formal" and args.max_iterations != 1000:
    parser.error("formal training requires exactly 1000 iterations")
RUN = args.run_root.resolve()
RUN.mkdir(parents=True, exist_ok=False)
if sha256_file(args.reference.resolve()) != args.reference_sha256:
    raise SystemExit("PRECONTACT_REFERENCE_SHA_MISMATCH")
write_json(RUN / "training_status.json", {"status": "STARTING", "phase": "APP_LAUNCH", "iteration": 0})
simulation_app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from agile.rl_env.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_agent_cfg import S203TPPORunnerCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env import S203TContactEnv  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env_cfg import S203TContactEnvCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_mdp import runtime_state  # noqa: E402


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


history: list[dict[str, object]] = []


class StatusRunner(OnPolicyRunner):
    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        metric = runtime_state(self.env.unwrapped).ensure()
        losses = {name: float(value) for name, value in locs["loss_dict"].items()}
        record = {
            "iteration": int(locs["it"]),
            "losses": losses,
            "losses_finite": all(math.isfinite(value) for value in losses.values()),
            "mean_surface_gap_m": float(metric["gaps"].mean()),
            "bilateral_contact_fraction": float(metric["contacts"].all(dim=-1).float().mean()),
            "success_fraction": float(metric["success"].float().mean()),
            "mean_box_translation_m": float(metric["box_translation"].mean()),
            "mean_root_height_m": float(metric["root_height"].mean()),
            "maximum_root_tilt_deg": float(metric["root_tilt_deg"].max()),
            "minimum_arm_joint_margin_rad": float(metric["arm_joint_margin"].min()),
            "maximum_arm_torque_ratio": float(metric["arm_torque_ratio"].max()),
            "wall_time_epoch_s": time.time(),
        }
        history.append(record)
        write_json(
            RUN / "training_status.json",
            {
                "status": "RUNNING",
                "phase": "LEARN",
                "iteration": int(locs["it"]) + 1,
                "target_iterations": args.max_iterations,
                "latest": record,
                "heartbeat_epoch_s": time.time(),
            },
        )


env = None
try:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg = S203TContactEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = S203TContactEnv(cfg=cfg)
    env.install_precontact_reference(args.reference.resolve())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    agent_cfg = S203TPPORunnerCfg()
    agent_cfg.seed = 42
    agent_cfg.device = args.device
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.resume = False
    agent_cfg.load_run = None
    agent_cfg.load_checkpoint = None
    agent_cfg.load_optimizer = False
    if agent_cfg.resume or agent_cfg.load_run is not None or agent_cfg.load_checkpoint is not None:
        raise RuntimeError("CLEAN_TRAINING_CONTRACT_VIOLATION")
    runner = StatusRunner(wrapped, agent_cfg.to_dict(), log_dir=str(RUN), device=agent_cfg.device)
    initial_state = runner.alg.policy.state_dict()
    actor_initial_sha = tensor_state_sha256(initial_state)
    initialization = {
        "schema_version": 1,
        "status": "PASS",
        "seed": 42,
        "actor_state_sha256_before_training": actor_initial_sha,
        "actor_input_dim": wrapped.num_obs,
        "actor_output_dim": wrapped.num_actions,
        "critic_input_dim": wrapped.num_obs,
        "clean_actor": True,
        "clean_optimizer": True,
        "clean_normalizer": True,
        "resume": False,
        "checkpoint_loaded": False,
        "pilot_checkpoint_used": False,
        "model_1999_used": False,
        "reference_path": str(args.reference.resolve()),
        "reference_sha256": args.reference_sha256,
        "lower_body_checkpoint_sha256": env.action_manager.get_term("frozen_lower_body").checkpoint_sha256,
        "num_envs": args.num_envs,
        "mode": args.mode,
    }
    if (wrapped.num_obs, wrapped.num_actions) != (77, 14):
        raise RuntimeError(f"RUNNER_DIMENSION_MISMATCH:{wrapped.num_obs}:{wrapped.num_actions}")
    write_json(RUN / "clean_initialization.json", initialization)
    write_json(
        RUN / "training_status.json",
        {"status": "RUNNING", "phase": "LEARN", "iteration": 0, "target_iterations": args.max_iterations},
    )
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    model_finite = all(bool(torch.isfinite(value).all()) for value in runner.alg.policy.state_dict().values())
    losses_finite = all(bool(item["losses_finite"]) for item in history)
    checkpoints = []
    for path in sorted(RUN.glob("model_*.pt"), key=lambda item: int(item.stem.split("_")[-1])):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        checkpoints.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "iteration_field": int(payload["iter"]),
                "actor_input_dim": 77,
                "actor_output_dim": 14,
            }
        )
    expected_final_iteration = args.max_iterations - 1
    final_candidates = [item for item in checkpoints if item["iteration_field"] == expected_final_iteration]
    complete = bool(final_candidates and model_finite and losses_finite and len(history) == args.max_iterations)
    result = {
        "schema_version": 1,
        "stage": "S2-03T",
        "mode": args.mode.upper(),
        "status": "PASS" if complete else "INVALID",
        "primary_reason": "TRAINING_COMPLETED_FINITE" if complete else "TRAINING_EVIDENCE_INCOMPLETE",
        "num_envs": args.num_envs,
        "requested_iterations": args.max_iterations,
        "completed_iterations": len(history),
        "runner_final_iteration": int(runner.current_learning_iteration),
        "finite": model_finite and losses_finite,
        "resume": False,
        "checkpoint_loaded": False,
        "pilot_checkpoint_used": False,
        "initial_actor_sha256": actor_initial_sha,
        "reference_sha256": args.reference_sha256,
        "checkpoints": checkpoints,
        "final_checkpoint": final_candidates[-1] if final_candidates else None,
        "curve": history,
    }
    write_json(RUN / "training_result.json", result)
    write_json(
        RUN / "training_status.json",
        {"status": "COMPLETE" if complete else "INVALID", "phase": "DONE", "iteration": len(history)},
    )
    if not complete:
        raise SystemExit(2)
except BaseException as exc:
    if not isinstance(exc, SystemExit) or exc.code != 2:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        if not (RUN / "training_result.json").is_file():
            write_json(
                RUN / "training_result.json",
                {
                    "schema_version": 1,
                    "stage": "S2-03T",
                    "mode": args.mode.upper(),
                    "status": "INVALID",
                    "primary_reason": "IMPLEMENTATION_EXCEPTION",
                    "error": repr(exc),
                    "num_envs": args.num_envs,
                    "requested_iterations": args.max_iterations,
                    "completed_iterations": len(history),
                },
            )
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
