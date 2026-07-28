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
parser.add_argument("--checkpoint-iteration", type=int)
parser.add_argument("--controller-id", choices=("actor", "original_s2_03"), default="actor")
parser.add_argument("--visual-evidence", action="store_true")
parser.add_argument(
    "--visual-evidence-id",
    choices=("CLEAN_UNTRAINED_ACTOR", "BEST_GAP_CHECKPOINT", "ORIGINAL_S2_03_CONTROLLER"),
)
parser.add_argument("--visual-frame-stride", type=int, default=2)
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
if args.controller_id == "original_s2_03" and checkpoint_path is not None:
    raise SystemExit("ORIGINAL_CONTROLLER_MUST_NOT_LOAD_ACTOR_CHECKPOINT")
if args.controller_id == "original_s2_03" and not args.visual_evidence:
    raise SystemExit("ORIGINAL_CONTROLLER_IS_VISUAL_DIAGNOSTIC_ONLY")
if args.visual_evidence and not args.enable_cameras:
    raise SystemExit("VISUAL_EVIDENCE_REQUIRES_ENABLE_CAMERAS")
if args.visual_evidence and args.visual_evidence_id is None:
    raise SystemExit("VISUAL_EVIDENCE_ID_REQUIRED")
if args.visual_evidence_id == "CLEAN_UNTRAINED_ACTOR" and checkpoint_path is not None:
    raise SystemExit("CLEAN_UNTRAINED_ACTOR_MUST_NOT_LOAD_CHECKPOINT")
if args.visual_evidence_id == "BEST_GAP_CHECKPOINT" and checkpoint_path is None:
    raise SystemExit("BEST_GAP_CHECKPOINT_REQUIRES_CHECKPOINT")
if args.visual_evidence_id == "ORIGINAL_S2_03_CONTROLLER" and args.controller_id != "original_s2_03":
    raise SystemExit("ORIGINAL_VISUAL_ID_REQUIRES_ORIGINAL_CONTROLLER")
if args.controller_id == "original_s2_03" and args.visual_evidence_id != "ORIGINAL_S2_03_CONTROLLER":
    raise SystemExit("ORIGINAL_CONTROLLER_VISUAL_ID_MISMATCH")
write_json(RUN / "runner_status.json", {"status": "STARTING", "phase": "APP_LAUNCH"})
simulation_app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from PIL import Image  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from agile.rl_env.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_agent_cfg import S203TPPORunnerCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_bootstrap import (  # noqa: E402
    MAXIMUM_ORIENTATION_CORRECTION_RAD,
    MAXIMUM_POSITION_CORRECTION_M,
    _clamp_norm,
    _target_pose_in_pelvis,
    derive_precontact_reference,
)
from g1_access_push.sim.stage2.s2_03t_env import S203TContactEnv  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_eval_cfg import S203TContactEvaluationEnvCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_mdp import runtime_state  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_visual_cfg import S203TVisualEvidenceEnvCfg  # noqa: E402
from g1_access_push.stage2.s2_03_contract import advance_rate_limited, load_config  # noqa: E402

from s2_03t_visual_recorder import VisualEvidenceRecorder  # noqa: E402

class VisualS203TContactEnv(S203TContactEnv):
    """Visual-only subclass that captures the terminal physics frame before auto-reset."""

    def __init__(self, *env_args, **env_kwargs) -> None:
        self.visual_terminal_callback = None
        super().__init__(*env_args, **env_kwargs)

    def _reset_idx(self, env_ids) -> None:
        if (
            self.visual_terminal_callback is not None
            and hasattr(self, "episode_length_buf")
            and not self._s2_03t_bootstrap_mode
        ):
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            completed = ids[self.episode_length_buf[ids] > 0]
            if bool((completed == 0).any()):
                self.visual_terminal_callback()
        super()._reset_idx(env_ids)



env = None
visual_recorder = None
try:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg = S203TVisualEvidenceEnvCfg() if args.visual_evidence else S203TContactEvaluationEnvCfg()
    cfg.seed = args.development_seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env_class = VisualS203TContactEnv if args.visual_evidence else S203TContactEnv
    render_mode = "rgb_array" if args.visual_evidence else None
    env = env_class(cfg=cfg, render_mode=render_mode)
    camera = env.scene["audit_camera"]
    camera.set_world_poses_from_view(
        torch.tensor([[-1.6, -2.2, 1.40]], device=env.device),
        torch.tensor([[0.45, 0.0, 0.70]], device=env.device),
    )
    if args.visual_evidence:
        side_camera = env.scene["side_camera"]
        side_camera.set_world_poses_from_view(
            torch.tensor([[0.55, -2.35, 0.92]], device=env.device),
            torch.tensor([[0.55, 0.0, 0.68]], device=env.device),
        )
    else:
        side_camera = None
    if args.visual_evidence:
        visual_recorder = VisualEvidenceRecorder(
            RUN,
            env,
            controller_id=args.visual_evidence_id,
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
            checkpoint_sha256=None if checkpoint_path is None else args.checkpoint_sha256,
            checkpoint_iteration=args.checkpoint_iteration,
            seed=args.development_seed,
            reference_path=str(reference_path),
            reference_sha256=args.reference_sha256,
            frame_stride=args.visual_frame_stride,
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

    def append_record(fsm_state: str, metric, *, terminal_snapshot=None) -> dict[str, object]:
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
            forbidden = bool(metric["failure_forbidden_non_palm_box_collision"][0])
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

        return record
    def bootstrap_record(phase: str, _step: int, metric) -> None:
        state_name = "STAND_SETTLE" if phase == "STAND_SETTLE" else "PRECONTACT"
        if records and records[-1]["fsm_state"] != state_name:
            transitions.append({"frame": len(records) - 1, "state": state_name, "reason": None})
        record = append_record(state_name, metric)
        if visual_recorder is not None:
            keyframe = "initial" if int(record["frame"]) == 0 else None
            visual_recorder.observe(record, keyframe=keyframe)

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
    write_json(
        RUN / "precontact_reference_replay_comparison.json",
        {
            "schema_version": 1,
            "status": "DIAGNOSTIC_ONLY",
            "camera_enabled_bootstrap": True,
            "tensor_max_abs_diff": tensor_diffs,
            "maximum_abs_diff": replay_max_diff,
            "scientific_gate": False,
        },
    )
    if not math.isfinite(replay_max_diff):
        raise RuntimeError("PRECONTACT_REFERENCE_REPLAY_NONFINITE")
    env.install_precontact_reference(stored_reference)
    env._s2_03t_terminal_snapshots.clear()
    initial_observation, _ = env.reset(seed=args.development_seed)
    installed_tensor_diffs = env.installed_reference_tensor_diffs(0)
    installed_max_diff = max(installed_tensor_diffs.values())
    arm = env.action_manager.get_term("arm_residual")
    lower = env.action_manager.get_term("frozen_lower_body")
    installed_state = runtime_state(env)
    installed_metric = installed_state.ensure()
    installed_checks = {
        "reference_tensors_finite": all(math.isfinite(value) for value in installed_tensor_diffs.values()),
        "reference_tensors_within_tolerance": installed_max_diff <= 1.0e-5,
        "arm_processed_equals_reference": bool(torch.equal(arm.processed_actions, arm.reference)),
        "arm_raw_history_zero": bool(torch.count_nonzero(arm.raw_actions) == 0),
        "arm_previous_history_zero": bool(torch.count_nonzero(arm.previous_raw_actions) == 0),
        "arm_delta_history_zero": bool(torch.count_nonzero(arm.action_delta) == 0),
        "arm_residual_zero": bool(torch.count_nonzero(arm.applied_residual) == 0),
        "lower_public_action_empty": bool(lower.raw_actions.numel() == 0),
        "lower_policy_matches_previous": bool(torch.equal(lower.policy_actions, lower.previous_policy_actions)),
        "lower_processed_target_finite": bool(torch.isfinite(lower.processed_actions).all()),
        "impulse_history_zero": bool(torch.count_nonzero(installed_state.impulse) == 0),
        "force_history_zero": bool(torch.count_nonzero(installed_state.previous_force) == 0),
        "contact_counters_zero": bool(
            torch.count_nonzero(installed_state.verify_count) == 0
            and torch.count_nonzero(installed_state.hold_count) == 0
            and torch.count_nonzero(installed_state.single_hand_count) == 0
            and torch.count_nonzero(installed_state.loss_streak) == 0
        ),
        "episode_length_zero": bool(torch.count_nonzero(env.episode_length_buf) == 0),
        "terminal_snapshots_clear": not bool(env._s2_03t_terminal_snapshots),
        "contacts_clear": not bool(installed_metric["contacts"].any()),
        "box_reference_zero": bool(torch.count_nonzero(installed_metric["box_translation"]) == 0),
        "base_reference_zero": bool(torch.count_nonzero(installed_metric["base_excursion"]) == 0),
        "initial_observation_shape": list(initial_observation["policy"].shape) == [1, 77],
        "initial_observation_finite": bool(torch.isfinite(initial_observation["policy"]).all()),
    }
    installed_valid = (
        math.isfinite(installed_max_diff)
        and installed_max_diff <= 1.0e-5
        and all(installed_checks.values())
    )
    write_json(
        RUN / "installed_reference_audit.json",
        {
            "schema_version": 1,
            "status": "PASS" if installed_valid else "INVALID",
            "reference_path": str(reference_path),
            "reference_sha256": args.reference_sha256,
            "tensor_max_abs_diff": installed_tensor_diffs,
            "maximum_abs_diff": installed_max_diff,
            "threshold": 1.0e-5,
            "checks": installed_checks,
        },
    )
    if not installed_valid:
        raise RuntimeError(f"INSTALLED_PRECONTACT_REFERENCE_MISMATCH:{installed_max_diff}")
    if visual_recorder is not None:
        precontact_visual_record = dict(records[-1])
        precontact_visual_record.update(
            {
                "left_actual_surface_gap_m": float(installed_metric["gaps"][0, 0]),
                "right_actual_surface_gap_m": float(installed_metric["gaps"][0, 1]),
                "left_contact": bool(installed_metric["contacts"][0, 0]),
                "right_contact": bool(installed_metric["contacts"][0, 1]),
                "left_force_n": float(installed_metric["forces"][0, 0]),
                "right_force_n": float(installed_metric["forces"][0, 1]),
                "root_tilt_deg": float(installed_metric["root_tilt_deg"][0]),
            }
        )
        visual_recorder.mark_precontact(precontact_visual_record)
    transitions.append({"frame": len(records) - 1, "state": "APPROACH_NORMAL", "reason": None})
    visual_terminal_captured = [False]

    def current_state_name() -> str:
        state = runtime_state(env)
        metric = state.ensure()
        if bool(state.contact_verified[0]):
            return "ATTACHED_HOLD" if int(state.hold_count[0]) > 0 else "BILATERAL_CONTACT_VERIFY"
        if bool(metric["contacts"][0].any()):
            return "BILATERAL_CONTACT_VERIFY"
        return "APPROACH_NORMAL"

    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    def capture_visual_terminal() -> None:
        if visual_recorder is None or visual_terminal_captured[0]:
            return
        metric = runtime_state(env).ensure()
        record = append_record(current_state_name(), metric)
        visual_recorder.observe(record, keyframe="terminal", actor_phase=True, force_video_frame=True)
        visual_terminal_captured[0] = True

    if visual_recorder is not None:
        env.visual_terminal_callback = capture_visual_terminal
    agent_cfg = S203TPPORunnerCfg()
    agent_cfg.resume = False
    agent_cfg.load_run = None
    agent_cfg.load_checkpoint = None
    agent_cfg.load_optimizer = False
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device) if args.controller_id == "actor" else None
    if checkpoint_path is not None:
        if runner is None:
            raise RuntimeError("ORIGINAL_CONTROLLER_RUNNER_MUST_BE_NONE")
        runner.load(str(checkpoint_path), load_optimizer=False)
    if runner is None:
        arm = env.action_manager.get_term("arm_residual")
        arm.set_bootstrap_mode(True)
        original_config = load_config(Path("/root/autodl-tmp/robotics/projects/g1_access_push/configs/stage2/s2_03_attach_only.yaml"))
        box_fixed_pos = env.scene["box"].data.root_link_pos_w.clone()
        box_fixed_quat = env.scene["box"].data.root_link_quat_w.clone()
        original_state = {"displacement": 0.0, "speed": 0.0, "acceleration": 0.0, "frozen": None}
        object_x = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)

        def original_policy(_observation):
            metric = runtime_state(env).ensure()
            if original_state["frozen"] is None and bool(metric["contacts"][0, 0] and metric["contacts"][0, 1]):
                original_state["frozen"] = original_state["displacement"]
            maximum_distance = float(original_config["motion"]["maximum_approach_distance_m"])
            if original_state["frozen"] is None and original_state["displacement"] < maximum_distance:
                displacement, speed, acceleration = advance_rate_limited(
                    original_state["displacement"], original_state["speed"], original_state["acceleration"],
                    dt_s=float(original_config["controller"]["control_dt_s"]),
                    speed_limit_mps=float(original_config["motion"]["approach_speed_mps"]),
                    acceleration_limit_mps2=float(original_config["motion"]["approach_acceleration_limit_mps2"]),
                    jerk_limit_mps3=float(original_config["motion"]["approach_jerk_limit_mps3"]),
                )
                original_state.update(displacement=min(displacement, maximum_distance), speed=speed, acceleration=acceleration)
            commanded = original_state["frozen"] if original_state["frozen"] is not None else original_state["displacement"]
            target_pos, target_quat = _target_pose_in_pelvis(env, box_fixed_pos, box_fixed_quat)
            _, object_quat_b = math_utils.subtract_frame_transforms(
                env.scene["robot"].data.root_link_pos_w,
                env.scene["robot"].data.root_link_quat_w,
                box_fixed_pos,
                box_fixed_quat,
            )
            object_x_b = math_utils.quat_apply(object_quat_b, object_x)
            desired_pos = target_pos + float(commanded) * object_x_b[:, None, :]
            palms = env.scene["hand_frames"]
            pos_error, orientation_error = math_utils.compute_pose_error(
                palms.data.target_pos_source.reshape(-1, 3),
                palms.data.target_quat_source.reshape(-1, 4),
                desired_pos.reshape(-1, 3),
                target_quat.reshape(-1, 4),
                rot_error_type="axis_angle",
            )
            commands = torch.cat(
                (
                    _clamp_norm(
                        pos_error.reshape(env.num_envs, 2, 3),
                        MAXIMUM_POSITION_CORRECTION_M,
                    ),
                    _clamp_norm(
                        orientation_error.reshape(env.num_envs, 2, 3),
                        MAXIMUM_ORIENTATION_CORRECTION_RAD,
                    ),
                ),
                dim=-1,
            )
            arm.set_bootstrap_commands(commands)
            if visual_recorder is not None:
                visual_recorder.set_original_commanded_normal_displacement(float(commanded))
            return torch.zeros((1, 14), device=env.device)

        policy = original_policy
    else:
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
            if visual_recorder is None:
                append_record(state_name, None, terminal_snapshot=terminal_snapshot)
            elif not visual_terminal_captured[0]:
                raise RuntimeError("VISUAL_TERMINAL_CAPTURE_MISSING")
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
        record = append_record(state_name, metric)
        if visual_recorder is not None:
            visual_recorder.observe(record, actor_phase=True)
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
            "controller_id": args.visual_evidence_id if args.visual_evidence else args.controller_id,
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "checkpoint_sha256": None if checkpoint_path is None else args.checkpoint_sha256,
            "checkpoint_iteration": args.checkpoint_iteration,
            "clean_untrained_actor": args.controller_id == "actor" and checkpoint_path is None,
            "deterministic_actor_mean": args.controller_id == "actor",
            "reference_path": str(reference_path),
            "reference_sha256": args.reference_sha256,
            "reference_replay_max_abs_diff": replay_max_diff,
            "reference_replay_gate": "DIAGNOSTIC_ONLY_CAMERA_ENABLED_BOOTSTRAP",
            "installed_reference_max_abs_diff": installed_max_diff,
            "installed_reference_tensor_diffs": installed_tensor_diffs,
            "reference_tensor_diffs": tensor_diffs,
            "precontact_replay_audit": replay_audit,
            "terminal_snapshot": terminal_snapshot,
            "post_initial_reset_count": 0,
            "box_push_commanded": False,
            "planner_started": False,
        },
    )
    if visual_recorder is not None:
        try:
            visual_recorder.finalize(
                records=records,
                terminal_state=terminal_state,
                failure_reason=failure_reason,
                result_primary_reason=failure_reason or "ALL_S2_03_GATES_PASSED",
                installed_reference_max_abs_diff=installed_max_diff,
                reference_replay_max_abs_diff=replay_max_diff,
            )
        except BaseException as visual_exc:
            print(f"VISUALIZATION_INVALID reason={visual_exc!r}", file=sys.stderr, flush=True)
            visual_recorder.close_incomplete(reason=f"VISUALIZATION_EXCEPTION:{visual_exc!r}")
    write_json(RUN / "runner_status.json", {"status": "COMPLETE", "phase": "EVIDENCE_READY"})
except BaseException as exc:
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    write_json(
        RUN / "runner_status.json",
        {"status": "INVALID", "primary_reason": "IMPLEMENTATION_EXCEPTION", "error": repr(exc)},
    )
    raise
finally:
    if visual_recorder is not None and not visual_recorder.finalized:
        visual_recorder.close_incomplete(reason="EVALUATOR_EXCEPTION_BEFORE_VISUAL_FINALIZE")
    if env is not None:
        env.close()
    simulation_app.close()
