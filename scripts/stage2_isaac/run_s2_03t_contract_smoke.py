#!/usr/bin/env python3
"""One-environment S2-03T reference extraction and runtime contract smoke."""

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
parser.add_argument("--contract-smoke", action="store_true", required=True)
parser.add_argument("--steps", type=int, default=3)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 2 <= args.steps <= 5:
    parser.error("--steps must be within [2,5]")
RUN = args.run_root.resolve()
RUN.mkdir(parents=True, exist_ok=False)
write_json(RUN / "runner_status.json", {"status": "STARTING", "phase": "APP_LAUNCH"})
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

from g1_access_push.sim.stage2.s2_03t_actions import ARM_JOINT_NAMES  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_bootstrap import derive_precontact_reference  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env import S203TContactEnv  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env_cfg import (  # noqa: E402
    CERTIFIED_STUDENT_SHA256,
    S203TContactEnvCfg,
)
from g1_access_push.sim.stage2.s2_03t_mdp import TERMINATION_METRIC_NAMES, runtime_state  # noqa: E402


env = None
try:
    write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": "ENVIRONMENT_CREATE"})
    cfg = S203TContactEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = S203TContactEnv(cfg=cfg)
    reference, reference_audit = derive_precontact_reference(env)
    reference_path = RUN / "precontact_reference.pt"
    torch.save(reference, reference_path)
    reference_sha = sha256_file(reference_path)
    reference_audit["reference_path"] = str(reference_path)
    reference_audit["reference_sha256"] = reference_sha
    write_json(RUN / "precontact_reference_audit.json", reference_audit)
    env.install_precontact_reference(reference)

    observation, _ = env.reset(seed=42)
    arm = env.action_manager.get_term("arm_residual")
    lower = env.action_manager.get_term("frozen_lower_body")
    state = runtime_state(env)
    sensor_left = env.scene["left_palm_box_contact"]
    sensor_right = env.scene["right_palm_box_contact"]
    action_names = list(env.action_manager.active_terms)
    action_dims = list(env.action_manager.action_term_dim)
    termination_names = set(env.termination_manager.active_terms)
    expected_termination_names = {"success", "time_out", *TERMINATION_METRIC_NAMES}
    checks = {
        "manager_based_rl_env": type(env).__name__ == "S203TContactEnv",
        "one_environment": env.num_envs == 1,
        "observation_shape_1x77": list(observation["policy"].shape) == [1, 77],
        "observation_finite_after_reset": bool(torch.isfinite(observation["policy"]).all()),
        "action_contract_14_plus_0": action_names == ["arm_residual", "frozen_lower_body"]
        and action_dims == [14, 0]
        and env.action_manager.total_action_dim == 14,
        "joint_order": tuple(arm.joint_names) == ARM_JOINT_NAMES,
        "certified_checkpoint_sha": lower.checkpoint_sha256 == CERTIFIED_STUDENT_SHA256,
        "no_actor_checkpoint_loaded": True,
        "left_sensor_initialized": sensor_left.is_initialized,
        "right_sensor_initialized": sensor_right.is_initialized,
        "separate_sensor_instances": sensor_left is not sensor_right,
        "left_filter_count_one": int(sensor_left.contact_physx_view.filter_count) == 1,
        "right_filter_count_one": int(sensor_right.contact_physx_view.filter_count) == 1,
        "termination_fields_complete": termination_names == expected_termination_names,
        "reference_installed": env.precontact_reference is not None,
        "arm_previous_action_zero": bool(torch.count_nonzero(arm.raw_actions) == 0),
        "impulse_history_zero": bool(torch.count_nonzero(state.impulse) == 0),
        "contact_counters_zero": bool(
            torch.count_nonzero(state.verify_count) == 0
            and torch.count_nonzero(state.hold_count) == 0
            and torch.count_nonzero(state.single_hand_count) == 0
        ),
    }
    actions = torch.zeros((1, 14), device=env.device)
    step_records = []
    for step in range(args.steps):
        observation, reward, terminated, truncated, _ = env.step(actions)
        metric = state.ensure()
        step_records.append(
            {
                "step": step + 1,
                "reward": float(reward[0]),
                "terminated": bool(terminated[0]),
                "truncated": bool(truncated[0]),
                "left_force_n": float(metric["forces"][0, 0]),
                "right_force_n": float(metric["forces"][0, 1]),
                "left_gap_m": float(metric["gaps"][0, 0]),
                "right_gap_m": float(metric["gaps"][0, 1]),
                "forbidden_collision": bool(metric["forbidden_non_palm_box_collision"][0]),
                "observation_finite": bool(torch.isfinite(observation["policy"]).all()),
            }
        )
        print(f"PHASE=CONTRACT_SMOKE step={step + 1}/{args.steps}", flush=True)
    checks.update(
        {
            "zero_step_reward_finite": all(torch.isfinite(torch.tensor(record["reward"])) for record in step_records),
            "zero_step_observation_finite": all(record["observation_finite"] for record in step_records),
            "zero_action_exact_reference": bool(torch.equal(arm.processed_actions, arm.reference)),
            "zero_action_no_done": not any(record["terminated"] or record["truncated"] for record in step_records),
            "zero_action_no_contact": not any(record["left_force_n"] >= 1.0 or record["right_force_n"] >= 1.0 for record in step_records),
            "zero_action_no_forbidden_collision": not any(record["forbidden_collision"] for record in step_records),
        }
    )
    env.reset(seed=42)
    checks.update(
        {
            "reset_arm_history_zero": bool(torch.count_nonzero(arm.raw_actions) == 0),
            "reset_impulse_history_zero": bool(torch.count_nonzero(state.impulse) == 0),
            "reset_contact_counters_zero": bool(
                torch.count_nonzero(state.verify_count) == 0
                and torch.count_nonzero(state.hold_count) == 0
                and torch.count_nonzero(state.single_hand_count) == 0
            ),
            "reset_lower_hidden_restored": bool(
                torch.equal(lower.hidden_state[0, 0].cpu(), reference["lower_hidden_state"])
            ),
            "reset_lower_cell_restored": bool(
                torch.equal(lower.cell_state[0, 0].cpu(), reference["lower_cell_state"])
            ),
            "reset_previous_lower_action_restored": bool(
                torch.equal(
                    lower.previous_policy_actions[0].cpu(), reference["previous_lower_policy_action"]
                )
            ),
        }
    )
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    result = {
        "schema_version": 1,
        "stage": "S2-03T",
        "mode": "CONTRACT_SMOKE",
        "status": "PASS" if not failed else "FAIL",
        "primary_reason": "ALL_RUNTIME_CONTRACT_CHECKS_PASSED" if not failed else "RUNTIME_CONTRACT_CHECK_FAILED",
        "checks": checks,
        "failed_checks": failed,
        "steps": step_records,
        "observation_shape": [1, 77],
        "action_shape": [1, 14],
        "reference_path": str(reference_path),
        "reference_sha256": reference_sha,
        "certified_checkpoint_sha256": lower.checkpoint_sha256,
        "box_mass_audit": env._s2_03t_mass_audit,
        "scientific_result_created": False,
    }
    write_json(RUN / "contract_smoke_result.json", result)
    write_json(RUN / "runner_status.json", {"status": "COMPLETE", "phase": "DONE", "result": result["status"]})
    if failed:
        raise SystemExit(2)
except BaseException as exc:
    if not isinstance(exc, SystemExit) or exc.code != 2:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        if not (RUN / "contract_smoke_result.json").is_file():
            write_json(
                RUN / "contract_smoke_result.json",
                {
                    "schema_version": 1,
                    "stage": "S2-03T",
                    "mode": "CONTRACT_SMOKE",
                    "status": "INVALID",
                    "primary_reason": "IMPLEMENTATION_EXCEPTION",
                    "error": repr(exc),
                    "scientific_result_created": False,
                },
            )
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
