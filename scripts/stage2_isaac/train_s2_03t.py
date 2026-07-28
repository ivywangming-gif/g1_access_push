#!/usr/bin/env python3
"""Clean, no-resume S2-03T redesign pilot or formal PPO training process."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PILOT_CONTRACT = {
    "num_envs": 64,
    "max_iterations": 200,
    "save_interval": 25,
    "curriculum_max_level": 1,
}
FORMAL_CONTRACT = {
    "num_envs": 256,
    "max_iterations": 1200,
    "save_interval": 100,
    "curriculum_max_level": 3,
}
EXPECTED_ACTOR_INPUT_DIM = 135
EXPECTED_CRITIC_INPUT_DIM = 175
EXPECTED_ACTION_DIM = 2
EXPECTED_ROLLOUT_STEPS_PER_ENV = 24
EXPECTED_CAMPAIGN = "S2_03T_REDESIGN_TRAIN_SCREEN_AND_VIDEO_CAMPAIGN"
REQUIRED_EXECUTION_SOURCES = (
    "scripts/stage2_isaac/train_s2_03t.py",
    "src/g1_access_push/sim/stage2/s2_03t_actions.py",
    "src/g1_access_push/sim/stage2/s2_03t_agent_cfg.py",
    "src/g1_access_push/sim/stage2/s2_03t_bootstrap.py",
    "src/g1_access_push/sim/stage2/s2_03t_env.py",
    "src/g1_access_push/sim/stage2/s2_03t_env_cfg.py",
    "src/g1_access_push/sim/stage2/s2_03t_mdp.py",
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"CAMPAIGN_MANIFEST_UNREADABLE:{path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("CAMPAIGN_MANIFEST_NOT_OBJECT")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def validate_campaign_manifest(
    path: Path,
    *,
    mode: str,
    run_root: Path,
    reference: Path,
    reference_sha256: str,
    invocation: Mapping[str, int],
) -> dict[str, Any]:
    payload = read_json_object(path)
    if payload.get("campaign") != EXPECTED_CAMPAIGN:
        raise ValueError("CAMPAIGN_MANIFEST_NAME_MISMATCH")
    manifest_root = Path(str(payload.get("run_root", ""))).resolve()
    expected_child_name = "pilot_training" if mode == "pilot" else "formal_training"
    if run_root.parent != manifest_root or run_root.name != expected_child_name:
        raise ValueError("CAMPAIGN_TRAINING_RUN_ROOT_MISMATCH")
    reference_record = payload.get("reference")
    if not isinstance(reference_record, dict):
        raise ValueError("CAMPAIGN_REFERENCE_RECORD_MISSING")
    if Path(str(reference_record.get("path", ""))).resolve() != reference:
        raise ValueError("CAMPAIGN_REFERENCE_PATH_MISMATCH")
    if reference_record.get("sha256") != reference_sha256:
        raise ValueError("CAMPAIGN_REFERENCE_SHA_MISMATCH")
    frozen = payload.get("frozen_execution")
    if not isinstance(frozen, dict):
        raise ValueError("CAMPAIGN_FROZEN_EXECUTION_MISSING")
    expected_frozen = {
        "action_dim": EXPECTED_ACTION_DIM,
        f"{mode}_env_count": invocation["num_envs"],
        f"{mode}_iterations": invocation["max_iterations"],
        f"{mode}_curriculum_max_level": invocation["curriculum_max_level"],
        "clean_actor": True,
        "resume": False,
        "model_1999_allowed": False,
        "pushing_enabled": False,
        "planner_enabled": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": frozen.get(key)}
        for key, expected in expected_frozen.items()
        if frozen.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"CAMPAIGN_FROZEN_EXECUTION_MISMATCH:{mismatches}")
    implementation_commit = payload.get("implementation_commit")
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise ValueError("CAMPAIGN_IMPLEMENTATION_COMMIT_MISSING")
    execution_sources = payload.get("execution_sources")
    if not isinstance(execution_sources, Mapping):
        raise ValueError("CAMPAIGN_EXECUTION_SOURCES_MISSING")
    if not set(REQUIRED_EXECUTION_SOURCES).issubset(execution_sources):
        raise ValueError("CAMPAIGN_REQUIRED_EXECUTION_SOURCE_MISSING")
    repository = Path(__file__).resolve().parents[2]
    for relative, record in execution_sources.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise ValueError("CAMPAIGN_EXECUTION_SOURCE_RECORD_INVALID")
        source = (repository / relative).resolve()
        if (
            Path(str(record.get("path", ""))).resolve() != source
            or not source.is_file()
            or sha256_file(source) != record.get("sha256")
        ):
            raise ValueError(f"CAMPAIGN_EXECUTION_SOURCE_SHA_MISMATCH:{relative}")
    return payload


parser = argparse.ArgumentParser()
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--reference-sha256", required=True)
parser.add_argument("--campaign-manifest", type=Path, required=True)
parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--max-iterations", type=int, required=True)
parser.add_argument("--save-interval", type=int, required=True)
parser.add_argument("--curriculum-max-level", type=int, required=True)
parser.add_argument("--seed", type=int, default=42)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seed != 42:
    parser.error("S2-03T clean actor seed is frozen at 42")
contract = PILOT_CONTRACT if args.mode == "pilot" else FORMAL_CONTRACT
invocation = {
    "num_envs": args.num_envs,
    "max_iterations": args.max_iterations,
    "save_interval": args.save_interval,
    "curriculum_max_level": args.curriculum_max_level,
}
if invocation != contract:
    parser.error(f"{args.mode} invocation differs from frozen contract: {invocation} != {contract}")
RUN = args.run_root.resolve()
REFERENCE = args.reference.resolve()
CAMPAIGN_MANIFEST = args.campaign_manifest.resolve()
manifest_payload = validate_campaign_manifest(
    CAMPAIGN_MANIFEST,
    mode=args.mode,
    run_root=RUN,
    reference=REFERENCE,
    reference_sha256=args.reference_sha256,
    invocation=invocation,
)
manifest_sha256 = sha256_file(CAMPAIGN_MANIFEST)
if not REFERENCE.is_file() or sha256_file(REFERENCE) != args.reference_sha256:
    raise SystemExit("PRECONTACT_REFERENCE_SHA_MISMATCH")
RUN.mkdir(parents=True, exist_ok=False)
write_json(
    RUN / "campaign_input.json",
    {
        "campaign_manifest": str(CAMPAIGN_MANIFEST),
        "campaign_manifest_sha256_at_start": manifest_sha256,
        "implementation_commit": manifest_payload["implementation_commit"],
        "mode": args.mode,
        "invocation": invocation,
        "reference": str(REFERENCE),
        "reference_sha256": args.reference_sha256,
    },
)
write_json(
    RUN / "training_status.json", {"status": "STARTING", "phase": "APP_LAUNCH", "iteration": 0}
)
simulation_app = AppLauncher(args).app

import torch  # noqa: E402
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from g1_access_push.sim.stage2.s2_03t_agent_cfg import S203TPPORunnerCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env import S203TContactEnv  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env_cfg import S203TContactEnvCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_mdp import (  # noqa: E402
    CONTACT_FORCE_THRESHOLD_N,
    SOFT_FORCE_MAXIMUM_N,
    runtime_state,
)


class TrainingMetricVecEnvWrapper(RslRlVecEnvWrapper):
    """Accumulate exact per-term rewards across every step in one PPO rollout."""

    def __init__(self, env: S203TContactEnv, clip_actions: float):
        super().__init__(env, clip_actions=clip_actions)
        self._reward_term_names = list(self.unwrapped.reward_manager.active_terms)
        self._reward_term_sum = torch.zeros(
            len(self._reward_term_names), dtype=torch.float64, device=self.device
        )
        self._reward_term_sample_count = 0

    def step(self, actions: torch.Tensor):
        result = super().step(actions)
        manager = self.unwrapped.reward_manager
        rates = manager._step_reward.detach()
        if rates.ndim != 2 or rates.shape[1] != len(self._reward_term_names):
            raise RuntimeError("TRAINING_REWARD_TERM_SHAPE_MISMATCH")
        contributions = rates * float(self.unwrapped.step_dt)
        self._reward_term_sum.add_(contributions.to(dtype=torch.float64).sum(dim=0))
        self._reward_term_sample_count += int(contributions.shape[0])
        return result

    def drain_reward_term_metrics(self) -> dict[str, object]:
        count = self._reward_term_sample_count
        expected_count = args.num_envs * EXPECTED_ROLLOUT_STEPS_PER_ENV
        if count != expected_count:
            raise RuntimeError(f"TRAINING_REWARD_INTERVAL_COUNT_MISMATCH:{count}:{expected_count}")
        step_mean = self._reward_term_sum / float(count)
        dt = float(self.unwrapped.step_dt)
        rate_mean = step_mean / dt
        result = {
            "reward_terms_mean": {
                name: float(step_mean[index].cpu())
                for index, name in enumerate(self._reward_term_names)
            },
            "reward_terms_weighted_rate_per_second": {
                name: float(rate_mean[index].cpu())
                for index, name in enumerate(self._reward_term_names)
            },
            "reward_total_step_mean": float(step_mean.sum().cpu()),
            "reward_term_names": list(self._reward_term_names),
            "reward_term_sample_count": count,
            "reward_metrics_source": "TrainingMetricVecEnvWrapper.rollout_accumulator",
        }
        self._reward_term_sum.zero_()
        self._reward_term_sample_count = 0
        return result


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported metric value: {type(value)!r}")


def fraction(mask: torch.Tensor) -> float:
    return float(mask.float().mean().detach().cpu())


def require_metric(metrics: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    value = metrics.get(name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"TRAINING_METRIC_MISSING:{name}")
    return value


def mask_union(metrics: Mapping[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    first = require_metric(metrics, names[0]).bool()
    combined = torch.zeros_like(first)
    for name in names:
        combined |= require_metric(metrics, name).bool()
    return combined


def gpu_memory_metrics(device: str) -> dict[str, object]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {"available": False, "device": device}
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    divisor = 1024.0 * 1024.0
    return {
        "available": True,
        "device": f"cuda:{device_index}",
        "allocated_mib": torch.cuda.memory_allocated(device_index) / divisor,
        "reserved_mib": torch.cuda.memory_reserved(device_index) / divisor,
        "maximum_allocated_mib": torch.cuda.max_memory_allocated(device_index) / divisor,
        "free_mib": free_bytes / divisor,
        "total_mib": total_bytes / divisor,
    }


def current_training_metrics(
    wrapper: TrainingMetricVecEnvWrapper, locs: Mapping[str, object]
) -> dict[str, object]:
    env = wrapper.unwrapped
    state = runtime_state(env)
    metric = state.ensure()
    contacts = require_metric(metric, "contacts").bool()
    gaps = require_metric(metric, "gaps")
    forces = require_metric(metric, "forces")
    bilateral = contacts.all(dim=-1)
    force_band = ((forces >= CONTACT_FORCE_THRESHOLD_N) & (forces <= SOFT_FORCE_MAXIMUM_N)).all(
        dim=-1
    )
    hard_force = mask_union(
        metric,
        (
            "failure_force_peak",
            "failure_palm_impulse",
            "failure_combined_impulse",
            "failure_force_rate",
        ),
    )
    pushing = mask_union(
        metric,
        (
            "failure_box_linear_speed",
            "failure_box_angular_speed",
            "failure_box_translation",
            "failure_box_yaw_change",
        ),
    )
    falling = mask_union(metric, ("failure_root_height", "failure_root_tilt"))
    hard_safety = mask_union(
        metric,
        (
            "failure_nonfinite",
            "failure_forbidden_non_palm_box_collision",
            "failure_force_peak",
            "failure_palm_impulse",
            "failure_combined_impulse",
            "failure_force_rate",
            "failure_base_excursion",
            "failure_root_height",
            "failure_root_tilt",
            "failure_arm_joint_margin",
            "failure_arm_torque",
        ),
    )
    raw_actions = state.arm.raw_actions
    action_delta = state.arm.action_delta
    rewbuffer = list(locs["rewbuffer"])
    lenbuffer = list(locs["lenbuffer"])
    if rewbuffer:
        episode_return = sum(float(item) for item in rewbuffer) / len(rewbuffer)
        episode_return_source = "completed_episode_rolling_100"
    else:
        partial = locs["cur_reward_sum"]
        if not isinstance(partial, torch.Tensor):
            raise RuntimeError("TRAINING_PARTIAL_RETURN_MISSING")
        episode_return = float(partial.mean().detach().cpu())
        episode_return_source = "partial_episode_mean_no_completed_episode_yet"
    sampled = env.sampled_initial_gap_m
    fallback = {
        "bilateral_contact_fraction": fraction(bilateral),
        "bilateral_contact_onset_fraction": fraction(
            require_metric(metric, "bilateral_contact_onset_event").bool()
        ),
        "verify_fraction": fraction(require_metric(metric, "bilateral_verify_step").bool()),
        "hold_success_fraction": fraction(require_metric(metric, "success").bool()),
        "hold_step_fraction": fraction(require_metric(metric, "attached_hold_step").bool()),
        "contact_retention_fraction": fraction(
            bilateral & (require_metric(metric, "reward_mode") == 2)
        ),
        "mean_surface_gap_m": float(gaps.mean().detach().cpu()),
        "minimum_surface_gap_m": float(gaps.min().detach().cpu()),
        "maximum_surface_gap_m": float(gaps.max().detach().cpu()),
        "surface_gap_distribution_m": {
            "left_mean": float(gaps[:, 0].mean().detach().cpu()),
            "right_mean": float(gaps[:, 1].mean().detach().cpu()),
            "mean": float(gaps.mean().detach().cpu()),
            "minimum": float(gaps.min().detach().cpu()),
            "maximum": float(gaps.max().detach().cpu()),
        },
        "sampled_initial_gap_distribution_m": {
            "mean": float(sampled.mean().detach().cpu()),
            "minimum": float(sampled.min().detach().cpu()),
            "maximum": float(sampled.max().detach().cpu()),
        },
        "force_band_occupancy_fraction": fraction(force_band),
        "hard_force_termination_fraction": fraction(hard_force),
        "hard_safety_violation_fraction": fraction(hard_safety),
        "pushing_fraction": fraction(pushing),
        "fall_fraction": fraction(falling),
        "maximum_root_tilt_deg": float(
            require_metric(metric, "root_tilt_deg").max().detach().cpu()
        ),
        "mean_root_tilt_deg": float(require_metric(metric, "root_tilt_deg").mean().detach().cpu()),
        "maximum_arm_torque_ratio": float(
            require_metric(metric, "arm_torque_ratio").max().detach().cpu()
        ),
        "minimum_arm_joint_margin_rad": float(
            require_metric(metric, "arm_joint_margin").min().detach().cpu()
        ),
        "action_saturation_fraction": fraction((raw_actions.abs() >= 0.999).any(dim=-1)),
        "mean_absolute_action_rate": float(action_delta.abs().mean().detach().cpu()),
        "maximum_absolute_action_rate": float(action_delta.abs().max().detach().cpu()),
        "episode_return_mean": episode_return,
        "episode_return_source": episode_return_source,
        "completed_episode_return_window_count": len(rewbuffer),
        "mean_episode_length_steps": (
            sum(float(item) for item in lenbuffer) / len(lenbuffer) if lenbuffer else 0.0
        ),
    }
    drain = getattr(state, "drain_training_interval", None)
    if callable(drain):
        drained = drain()
        if not isinstance(drained, Mapping):
            raise RuntimeError("TRAINING_INTERVAL_METRICS_NOT_MAPPING")
        interval = json_safe(drained)
        if not isinstance(interval, dict):
            raise RuntimeError("TRAINING_INTERVAL_METRICS_NOT_OBJECT")
        expected_samples = float(args.num_envs * EXPECTED_ROLLOUT_STEPS_PER_ENV)
        if float(interval.get("sample_count", -1.0)) != expected_samples:
            raise RuntimeError(
                f"TRAINING_INTERVAL_COUNT_MISMATCH:{interval.get('sample_count')}:{expected_samples}"
            )
        fallback.update(interval)
        fallback["metrics_source"] = "runtime_state.drain_training_interval"
    else:
        fallback["metrics_source"] = "current_state_fallback_no_interval_api"
    fallback.update(wrapper.drain_reward_term_metrics())
    fallback["surface_gap_distribution_m"] = {
        "mean": float(fallback["mean_surface_gap_m"]),
        "minimum": float(fallback["minimum_surface_gap_m"]),
        "maximum": float(fallback["maximum_surface_gap_m"]),
        "left_mean_at_iteration_end": float(gaps[:, 0].mean().detach().cpu()),
        "right_mean_at_iteration_end": float(gaps[:, 1].mean().detach().cpu()),
        "aggregate_source": fallback["metrics_source"],
    }
    if "mean_action_rate" in fallback:
        fallback["mean_absolute_action_rate"] = float(fallback["mean_action_rate"])
    if "maximum_action_rate" in fallback:
        fallback["maximum_absolute_action_rate"] = float(fallback["maximum_action_rate"])
    fallback["curriculum_level"] = int(env.curriculum_level)
    fallback["curriculum_maximum_level"] = args.curriculum_max_level
    fallback["curriculum_completed_episode_window"] = len(env._s2_03t_curriculum_window)
    return fallback


history: list[dict[str, object]] = []
checkpoint_records: dict[str, dict[str, object]] = {}
training_started_monotonic = time.monotonic()


class StatusRunner(OnPolicyRunner):
    def __init__(self, *runner_args, **runner_kwargs):
        super().__init__(*runner_args, **runner_kwargs)
        self._promotion_count = 0

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        losses = {name: float(value) for name, value in locs["loss_dict"].items()}
        iteration = int(locs["it"])
        env = self.env.unwrapped
        metrics = current_training_metrics(self.env, locs)
        promotions = env.curriculum_promotions
        new_promotions = promotions[self._promotion_count :]
        self._promotion_count = len(promotions)
        record = {
            "iteration": iteration,
            "losses": losses,
            "losses_finite": all(math.isfinite(value) for value in losses.values()),
            **metrics,
            "curriculum_promotions": promotions,
            "new_curriculum_promotions": new_promotions,
            "iteration_time_s": float(locs["collection_time"] + locs["learn_time"]),
            "collection_time_s": float(locs["collection_time"]),
            "learning_time_s": float(locs["learn_time"]),
            "gpu_memory": gpu_memory_metrics(self.device),
            "checkpoint_saved_after_rollout_update": None,
            "wall_time_epoch_s": time.time(),
        }
        if not finite_tree(record):
            raise RuntimeError(f"NONFINITE_TRAINING_CURVE_AT_ITERATION:{iteration}")
        history.append(record)
        append_jsonl(RUN / "training_curve.jsonl", record)
        write_json(
            RUN / "training_status.json",
            {
                "status": "RUNNING",
                "phase": "LEARN",
                "iteration": iteration + 1,
                "target_iterations": args.max_iterations,
                "latest": record,
                "heartbeat_epoch_s": time.time(),
            },
        )

    def save(self, path: str, infos=None):
        super().save(path, infos=infos)
        checkpoint_path = Path(path).resolve()
        serialized = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        filename_iteration = int(checkpoint_path.stem.rsplit("_", 1)[1])
        serialized_iteration = int(serialized.get("iter", -1))
        callback_iteration = int(self.current_learning_iteration)
        if checkpoint_path.name == "model_1999.pt":
            raise RuntimeError("MODEL_1999_FORBIDDEN")
        if not (filename_iteration == serialized_iteration == callback_iteration):
            raise RuntimeError(
                f"CHECKPOINT_ITERATION_MISMATCH:{filename_iteration}:"
                f"{serialized_iteration}:{callback_iteration}"
            )
        metadata = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "iteration_field": int(self.current_learning_iteration),
            "serialized_iteration": serialized_iteration,
            "optimizer_updates_completed": callback_iteration + 1,
            "saved_after_update": True,
            "behavior_metric_iteration": (
                callback_iteration + 1 if callback_iteration + 1 < args.max_iterations else None
            ),
            "behavior_metric_status": (
                "NEXT_ROLLOUT_MATCHES_SAVED_POLICY"
                if callback_iteration + 1 < args.max_iterations
                else "POST_UPDATE_BEHAVIOR_METRIC_MISSING_FINAL_FORCED_SCREENING"
            ),
            "actor_input_dim": EXPECTED_ACTOR_INPUT_DIM,
            "critic_input_dim": EXPECTED_CRITIC_INPUT_DIM,
            "actor_output_dim": EXPECTED_ACTION_DIM,
        }
        checkpoint_records[str(checkpoint_path)] = metadata
        for record in reversed(history):
            if int(record["iteration"]) == int(self.current_learning_iteration):
                record["checkpoint_saved_after_rollout_update"] = metadata
                break
        append_jsonl(
            RUN / "checkpoint_manifest.jsonl",
            {"event": "CHECKPOINT_SAVED", "wall_time_epoch_s": time.time(), **metadata},
        )
        write_json(
            RUN / "training_status.json",
            {
                "status": "RUNNING",
                "phase": "CHECKPOINT_SAVED",
                "iteration": int(self.current_learning_iteration) + 1,
                "target_iterations": args.max_iterations,
                "latest_checkpoint": metadata,
                "heartbeat_epoch_s": time.time(),
            },
        )


env = None
try:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = S203TContactEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = S203TContactEnv(cfg=cfg)
    env.configure_contact_curriculum(enabled=True, maximum_level=args.curriculum_max_level)
    env.install_precontact_reference(REFERENCE)
    wrapped = TrainingMetricVecEnvWrapper(env, clip_actions=1.0)
    dimensions = {
        "actor_input_dim": int(wrapped.num_obs),
        "critic_input_dim": int(wrapped.num_privileged_obs),
        "actor_output_dim": int(wrapped.num_actions),
    }
    expected_dimensions = {
        "actor_input_dim": EXPECTED_ACTOR_INPUT_DIM,
        "critic_input_dim": EXPECTED_CRITIC_INPUT_DIM,
        "actor_output_dim": EXPECTED_ACTION_DIM,
    }
    if dimensions != expected_dimensions:
        raise RuntimeError(f"RUNNER_DIMENSION_MISMATCH:{dimensions}:{expected_dimensions}")
    agent_cfg = S203TPPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.resume = False
    agent_cfg.load_run = None
    agent_cfg.load_checkpoint = None
    agent_cfg.load_optimizer = False
    if agent_cfg.resume or agent_cfg.load_run is not None or agent_cfg.load_checkpoint is not None:
        raise RuntimeError("CLEAN_TRAINING_CONTRACT_VIOLATION")
    resolved_training_config_path = RUN / "resolved_training_config.json"
    write_json(
        resolved_training_config_path,
        {
            "schema_version": 1,
            "mode": args.mode,
            "seed": args.seed,
            "invocation": invocation,
            "dimensions": dimensions,
            "runner_cfg": json_safe(agent_cfg.to_dict()),
        },
    )
    runner = StatusRunner(wrapped, agent_cfg.to_dict(), log_dir=str(RUN), device=agent_cfg.device)
    initial_state = runner.alg.policy.state_dict()
    initial_policy_sha = tensor_state_sha256(initial_state)
    initialization = {
        "schema_version": 2,
        "status": "PASS",
        "seed": args.seed,
        "policy_state_sha256_before_training": initial_policy_sha,
        **dimensions,
        "clean_actor": True,
        "clean_optimizer": True,
        "clean_normalizer": True,
        "resume": False,
        "checkpoint_loaded": False,
        "pilot_checkpoint_used": False,
        "model_1999_used": False,
        "reference_path": str(REFERENCE),
        "reference_sha256": args.reference_sha256,
        "campaign_manifest": str(CAMPAIGN_MANIFEST),
        "campaign_manifest_sha256_at_start": manifest_sha256,
        "implementation_commit": manifest_payload["implementation_commit"],
        "lower_body_checkpoint_sha256": env.action_manager.get_term(
            "frozen_lower_body"
        ).checkpoint_sha256,
        "num_envs": args.num_envs,
        "mode": args.mode,
        "maximum_curriculum_level": args.curriculum_max_level,
    }
    write_json(RUN / "clean_initialization.json", initialization)
    write_json(
        RUN / "training_status.json",
        {
            "status": "RUNNING",
            "phase": "LEARN",
            "iteration": 0,
            "target_iterations": args.max_iterations,
        },
    )
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    model_finite = all(
        bool(torch.isfinite(value).all()) for value in runner.alg.policy.state_dict().values()
    )
    losses_finite = all(bool(item["losses_finite"]) for item in history)
    runtime_nonfinite_zero = bool(
        history
        and all(float(item.get("nonfinite_fraction", float("nan"))) == 0.0 for item in history)
    )
    checkpoints = sorted(checkpoint_records.values(), key=lambda item: int(item["iteration_field"]))
    expected_final_iteration = args.max_iterations - 1
    final_candidates = [
        item for item in checkpoints if int(item["iteration_field"]) == expected_final_iteration
    ]
    checkpoints_valid = all(
        Path(str(item["path"])).is_file()
        and sha256_file(Path(str(item["path"]))) == item["sha256"]
        and Path(str(item["path"])).name != "model_1999.pt"
        for item in checkpoints
    )
    complete = bool(
        final_candidates
        and checkpoints_valid
        and model_finite
        and losses_finite
        and runtime_nonfinite_zero
        and len(history) == args.max_iterations
        and finite_tree(history)
    )
    result = {
        "schema_version": 2,
        "stage": "S2-03T",
        "training_design": "HYBRID_NORMAL_ACTION_THREE_MODE_REWARD_GAP_CURRICULUM",
        "mode": args.mode.upper(),
        "seed": args.seed,
        "clean_actor": True,
        "clean_optimizer": True,
        "clean_normalizer": True,
        "status": "PASS" if complete else "INVALID",
        "primary_reason": "TRAINING_COMPLETED_FINITE"
        if complete
        else "TRAINING_EVIDENCE_INCOMPLETE",
        "authoritative_complete_before_teardown": complete,
        "num_envs": args.num_envs,
        "requested_iterations": args.max_iterations,
        "completed_iterations": len(history),
        "runner_final_iteration": int(runner.current_learning_iteration),
        "save_interval": args.save_interval,
        "finite": model_finite and losses_finite and finite_tree(history),
        "runtime_nonfinite_fraction_zero": runtime_nonfinite_zero,
        "resolved_training_config": {
            "path": str(resolved_training_config_path),
            "sha256": sha256_file(resolved_training_config_path),
        },
        "dimensions": dimensions,
        "resume": False,
        "checkpoint_loaded": False,
        "pilot_checkpoint_used": False,
        "model_1999_used": False,
        "initial_policy_sha256": initial_policy_sha,
        "reference_path": str(REFERENCE),
        "reference_sha256": args.reference_sha256,
        "campaign_manifest": str(CAMPAIGN_MANIFEST),
        "campaign_manifest_sha256_at_start": manifest_sha256,
        "implementation_commit": manifest_payload["implementation_commit"],
        "curriculum": {
            "enabled": True,
            "maximum_level": args.curriculum_max_level,
            "final_level": int(env.curriculum_level),
            "promotions": env.curriculum_promotions,
        },
        "reward_term_names": list(env.reward_manager.active_terms),
        "checkpoints": checkpoints,
        "final_checkpoint": final_candidates[-1] if final_candidates else None,
        "curve": history,
        "wall_time_seconds": time.monotonic() - training_started_monotonic,
    }
    write_json(RUN / "training_result.json", result)
    write_json(
        RUN / "training_status.json",
        {
            "status": "COMPLETE" if complete else "INVALID",
            "phase": "DONE",
            "iteration": len(history),
            "final_checkpoint": result["final_checkpoint"],
        },
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
                    "schema_version": 2,
                    "stage": "S2-03T",
                    "mode": args.mode.upper(),
                    "status": "INVALID",
                    "primary_reason": "IMPLEMENTATION_EXCEPTION",
                    "authoritative_complete_before_teardown": False,
                    "error": repr(exc),
                    "num_envs": args.num_envs,
                    "requested_iterations": args.max_iterations,
                    "completed_iterations": len(history),
                    "resume": False,
                    "checkpoint_loaded": False,
                    "model_1999_used": False,
                },
            )
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
