#!/usr/bin/env python3
"""Actor-aware one-episode S2-03T evaluator that emits original S2-03 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
parser.add_argument("--checkpoint", default="NONE")
parser.add_argument("--checkpoint-sha256", default="NONE")
parser.add_argument("--mode", choices=("baseline", "pilot", "screening", "formal", "preflight"), required=True)
parser.add_argument("--development-seed", type=int, required=True)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
RUN = args.run_root.resolve()
RUN.mkdir(parents=True, exist_ok=False)
reference_path = args.reference.resolve()
if sha256_file(reference_path) != args.reference_sha256:
    raise SystemExit("PRECONTACT_REFERENCE_SHA_MISMATCH")
checkpoint_path = None if args.checkpoint == "NONE" else Path(args.checkpoint).resolve()
if checkpoint_path is not None:
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != args.checkpoint_sha256:
        raise SystemExit("ACTOR_CHECKPOINT_SHA_MISMATCH")
write_json(RUN / "runner_status.json", {"status": "STARTING", "phase": "APP_LAUNCH"})
simulation_app = AppLauncher(args).app

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from agile.rl_env.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_agent_cfg import S203TPPORunnerCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_bootstrap import derive_precontact_reference  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env import S203TContactEnv  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_eval_cfg import S203TContactEvaluationEnvCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_mdp import runtime_state  # noqa: E402


env = None
try:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg = S203TContactEvaluationEnvCfg()
    cfg.seed = args.development_seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = S203TContactEnv(cfg=cfg)
    camera = env.scene["audit_camera"]
    camera.set_world_poses_from_view(
        torch.tensor([[2.0, -2.0, 1.55]], device=env.device),
        torch.tensor([[0.45, 0.0, 0.70]], device=env.device),
    )
    records: list[dict[str, object]] = []
    transitions = [
        {"frame": -1, "state": "RESET", "reason": None},
        {"frame": -1, "state": "STAND_SETTLE", "reason": None},
    ]
    left_onset = right_onset = bilateral_onset = None
    left_losses: list[int] = []
    right_losses: list[int] = []
    previous_contacts = [False, False]

    def append_record(fsm_state: str, metric, *, terminal_snapshot=None) -> None:
        global left_onset, right_onset, bilateral_onset, previous_contacts
        frame = len(records)
        if terminal_snapshot is None:
            forces = [float(metric["forces"][0, 0]), float(metric["forces"][0, 1])]
            contacts = [bool(metric["contacts"][0, 0]), bool(metric["contacts"][0, 1])]
            gaps = [float(metric["gaps"][0, 0]), float(metric["gaps"][0, 1])]
            impulse = [float(value) for value in runtime_state(env).impulse[0]]
            force_rate = float(metric["force_rate"][0])
            box_translation = float(metric["box_translation"][0])
            box_yaw = float(metric["box_yaw_change"][0].abs())
            box_linear = float(metric["box_linear_speed"][0])
            box_angular = float(metric["box_angular_speed"][0])
            root_height = float(metric["root_height"][0])
            root_tilt = float(metric["root_tilt_deg"][0])
            joint_margin = float(metric["arm_joint_margin"][0])
            torque_ratio = float(metric["arm_torque_ratio"][0])
            finite = bool(metric["finite"][0])
            forbidden = bool(metric["forbidden_non_palm_box_collision"][0])
            position_error = metric["palm_position_error"][0]
            orientation_error = metric["palm_orientation_error"][0]
        else:
            forces = terminal_snapshot["maximum_force_n"]
            contacts = [terminal_snapshot["bilateral_contact_seen"]] * 2 if terminal_snapshot["success"] else list(previous_contacts)
            gaps = terminal_snapshot["minimum_surface_gap_m"]
            impulse = terminal_snapshot["impulse_ns"]
            force_rate = 0.0
            box_translation = terminal_snapshot["maximum_box_translation_m"]
            box_yaw = terminal_snapshot["maximum_box_yaw_change_rad"]
            box_linear = 0.0
            box_angular = 0.0
            root_height = terminal_snapshot["minimum_root_height_m"]
            root_tilt = terminal_snapshot["maximum_root_tilt_deg"]
            joint_margin = terminal_snapshot["minimum_arm_joint_margin_rad"]
            torque_ratio = terminal_snapshot["maximum_arm_torque_ratio"]
            finite = terminal_snapshot["finite"]
            forbidden = "forbidden_non_palm_box_collision" in terminal_snapshot["termination_reasons"]
            position_error = torch.zeros((2, 3), device=env.device)
            orientation_error = torch.zeros((2, 3), device=env.device)
        if contacts[0] and left_onset is None:
            left_onset = frame
        if contacts[1] and right_onset is None:
            right_onset = frame
        if contacts[0] and contacts[1] and bilateral_onset is None:
            bilateral_onset = frame
        if previous_contacts[0] and not contacts[0]:
            left_losses.append(frame)
        if previous_contacts[1] and not contacts[1]:
            right_losses.append(frame)
        previous_contacts = contacts
        record = {
            "frame": frame,
            "time_s": (frame + 1) * 0.02,
            "fsm_state": fsm_state,
            "finite": finite,
            "left_contact": contacts[0],
            "right_contact": contacts[1],
            "left_force_n": forces[0],
            "right_force_n": forces[1],
            "left_impulse_ns": impulse[0],
            "right_impulse_ns": impulse[1],
            "combined_impulse_ns": impulse[0] + impulse[1],
            "contact_force_peak_n": max(forces),
            "contact_force_rate_nps": force_rate,
            "left_contact_onset_frame": left_onset,
            "right_contact_onset_frame": right_onset,
            "bilateral_contact_onset_frame": bilateral_onset,
            "left_contact_loss_frames": list(left_losses),
            "right_contact_loss_frames": list(right_losses),
            "forbidden_contact_links": ["FORBIDDEN_NON_PALM_BOX_COLLISION"] if forbidden else [],
            "box_translation_m": box_translation,
            "box_yaw_change_rad": box_yaw,
            "box_linear_speed_mps": box_linear,
            "box_angular_speed_radps": box_angular,
            "root_height_m": root_height,
            "root_tilt_deg": root_tilt,
            "minimum_arm_joint_limit_margin_rad": joint_margin,
            "arm_torque_ratio_max": torque_ratio,
            "post_initial_reset_count": 0,
            "left_actual_surface_gap_m": gaps[0],
            "right_actual_surface_gap_m": gaps[1],
            "left_position_error_m": float(torch.linalg.vector_norm(position_error[0])),
            "right_position_error_m": float(torch.linalg.vector_norm(position_error[1])),
            "left_orientation_error_deg": math.degrees(float(torch.linalg.vector_norm(orientation_error[0]))),
            "right_orientation_error_deg": math.degrees(float(torch.linalg.vector_norm(orientation_error[1]))),
            "box_push_commanded": False,
        }
        records.append(record)

    def bootstrap_record(phase: str, _step: int, metric) -> None:
        state_name = "STAND_SETTLE" if phase == "STAND_SETTLE" else "PRECONTACT"
        if records and records[-1]["fsm_state"] != state_name:
            transitions.append({"frame": len(records) - 1, "state": state_name, "reason": None})
        append_record(state_name, metric)

    replay_reference, replay_audit = derive_precontact_reference(env, record_callback=bootstrap_record)
    stored_reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    tensor_diffs = {}
    for name in (
        "robot_root_state_relative",
        "robot_joint_position",
        "robot_joint_velocity",
        "box_root_state_relative",
        "arm_ik_target",
        "lower_hidden_state",
        "lower_cell_state",
        "previous_lower_policy_action",
    ):
        tensor_diffs[name] = float(torch.max(torch.abs(replay_reference[name] - stored_reference[name])))
    replay_max_diff = max(tensor_diffs.values())
    if replay_max_diff > 1.0e-5:
        raise RuntimeError(f"PRECONTACT_REFERENCE_REPLAY_MISMATCH:{replay_max_diff}")
    env.install_precontact_reference(stored_reference)
    env._s2_03t_terminal_snapshots.clear()
    env.reset(seed=args.development_seed)
    transitions.append({"frame": len(records) - 1, "state": "APPROACH_NORMAL", "reason": None})
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    agent_cfg = S203TPPORunnerCfg()
    agent_cfg.resume = False
    agent_cfg.load_run = None
    agent_cfg.load_checkpoint = None
    agent_cfg.load_optimizer = False
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if checkpoint_path is not None:
        runner.load(str(checkpoint_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    observation, _ = wrapped.get_observations()
    terminal_snapshot = None
    last_state_name = "APPROACH_NORMAL"
    for step in range(1000):
        with torch.inference_mode():
            action = policy(observation)
        if tuple(action.shape) != (1, 14) or not bool(torch.isfinite(action).all()):
            raise RuntimeError("ACTOR_ACTION_INVALID")
        observation, reward, done, _ = wrapped.step(action)
        if not bool(torch.isfinite(observation).all() and torch.isfinite(reward).all()):
            raise RuntimeError("EVALUATION_NONFINITE")
        if bool(done[0]):
            terminal_snapshot = env.pop_terminal_snapshot(0)
            if terminal_snapshot is None:
                raise RuntimeError("TERMINAL_SNAPSHOT_MISSING")
            state_name = "ATTACHED_HOLD" if terminal_snapshot["success"] else last_state_name
            append_record(state_name, None, terminal_snapshot=terminal_snapshot)
            break
        state = runtime_state(env)
        metric = state.ensure()
        if bool(state.contact_verified[0]):
            state_name = "ATTACHED_HOLD" if int(state.hold_count[0]) > 0 else "BILATERAL_CONTACT_VERIFY"
        elif bool(metric["contacts"][0].any()):
            state_name = "BILATERAL_CONTACT_VERIFY"
        else:
            state_name = "APPROACH_NORMAL"
        if state_name != last_state_name:
            transitions.append({"frame": len(records) - 1, "state": state_name, "reason": None})
            last_state_name = state_name
        append_record(state_name, metric)
        if (step + 1) % 25 == 0:
            print(f"PHASE=ACTOR_EVALUATION step={step + 1}", flush=True)
    if terminal_snapshot is None:
        raise RuntimeError("EVALUATION_TERMINAL_EVIDENCE_MISSING")

    reason_map = {
        "forbidden_non_palm_box_collision": "FORBIDDEN_BODY_BOX_COLLISION",
        "force_peak": "EXCESSIVE_CONTACT_IMPULSE",
        "palm_impulse": "EXCESSIVE_CONTACT_IMPULSE",
        "combined_impulse": "EXCESSIVE_CONTACT_IMPULSE",
        "force_rate": "EXCESSIVE_CONTACT_IMPULSE",
        "box_linear_speed": "OBJECT_TRANSLATION_DURING_ATTACH",
        "box_translation": "OBJECT_TRANSLATION_DURING_ATTACH",
        "box_angular_speed": "OBJECT_YAW_DURING_ATTACH",
        "box_yaw_change": "OBJECT_YAW_DURING_ATTACH",
        "base_excursion": "ROBOT_FALL",
        "root_height": "ROBOT_FALL",
        "root_tilt": "ROBOT_BAD_TILT",
        "arm_joint_margin": "JOINT_LIMIT_VIOLATION",
        "arm_torque": "TORQUE_LIMIT_VIOLATION",
        "nonfinite": "NONFINITE",
        "contact_loss": "LEFT_CONTACT_LOST",
        "single_hand_timeout": "BILATERAL_CONTACT_TIMEOUT",
    }
    if terminal_snapshot["success"]:
        terminal_state = "PASS"
        failure_reason = None
    else:
        terminal_state = "FAIL"
        reasons = terminal_snapshot["termination_reasons"]
        if terminal_snapshot["time_out"] and not reasons:
            failure_reason = "BILATERAL_CONTACT_TIMEOUT"
        elif reasons:
            failure_reason = reason_map[reasons[0]]
        else:
            failure_reason = "BILATERAL_CONTACT_TIMEOUT"
        transitions.append({"frame": len(records) - 1, "state": "FAIL", "reason": failure_reason})
    outcome = {
        "schema_version": 1,
        "stage": "S2-03",
        "terminal_state": terminal_state,
        "failure_reason": failure_reason,
        "fsm_transitions": transitions,
        "left_contact_onset_frame": left_onset,
        "right_contact_onset_frame": right_onset,
        "bilateral_contact_onset_frame": bilateral_onset,
        "left_contact_loss_frames": left_losses,
        "right_contact_loss_frames": right_losses,
        "left_impulse_ns": terminal_snapshot["impulse_ns"][0],
        "right_impulse_ns": terminal_snapshot["impulse_ns"][1],
        "observed_frames": len(records),
        "box_push_commanded": False,
    }
    with (RUN / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(RUN / "fsm_outcome.json", outcome)
    image = Image.fromarray(camera.data.output["rgb"][0].detach().cpu().numpy().astype("uint8"))
    image.save(RUN / "final.png")
    if terminal_state == "FAIL":
        image.save(RUN / "failure.png")
    write_json(
        RUN / "actor_evaluation_metadata.json",
        {
            "schema_version": 1,
            "stage": "S2-03T",
            "mode": args.mode.upper(),
            "development_seed": args.development_seed,
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "checkpoint_sha256": None if checkpoint_path is None else args.checkpoint_sha256,
            "clean_untrained_actor": checkpoint_path is None,
            "deterministic_actor_mean": True,
            "reference_path": str(reference_path),
            "reference_sha256": args.reference_sha256,
            "reference_replay_max_abs_diff": replay_max_diff,
            "reference_tensor_diffs": tensor_diffs,
            "precontact_replay_audit": replay_audit,
            "terminal_snapshot": terminal_snapshot,
            "post_initial_reset_count": 0,
            "box_push_commanded": False,
            "planner_started": False,
        },
    )
    write_json(RUN / "runner_status.json", {"status": "COMPLETE", "phase": "EVIDENCE_READY"})
except BaseException as exc:
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    write_json(
        RUN / "runner_status.json",
        {"status": "INVALID", "primary_reason": "IMPLEMENTATION_EXCEPTION", "error": repr(exc)},
    )
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
