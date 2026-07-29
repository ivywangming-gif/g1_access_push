#!/usr/bin/env python3
"""Run the isolated S2-03T pelvis contact-pair/path-recovery campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np

BOOT = argparse.ArgumentParser(add_help=False)
BOOT.add_argument("--run-root", type=Path, required=True)
boot_args, _ = BOOT.parse_known_args()
RUN = boot_args.run_root.resolve()
RUN.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


write_json(RUN / "runner_status.json", {"status": "STARTING", "phase": "APP_LAUNCH", "stage": "S2_03T_PELVIS_CONTACT_PAIR_AND_LEFT_ARM_PATH_RECOVERY"})

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--run-root", type=Path, required=True)
PARSER.add_argument("--reference", type=Path)
PARSER.add_argument("--seed", type=int, default=42)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
if ARGS.seed != 42:
    raise SystemExit("S2_03T_PATH_RECOVERY_SEED_MUST_BE_42")
SIMULATION_APP = AppLauncher(ARGS).app

import torch  # noqa: E402
from pxr import Usd  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

from g1_access_push.sim.stage2.s2_03t_safe_chest_joint_reference_env_cfg import S203TSafeChestJointReferenceEnvCfg  # noqa: E402
from g1_access_push.stage2.s2_03t_pelvis_contact_path_recovery_contract import (  # noqa: E402
    ARM_JOINT_NAMES,
    CONTROL_DT_S,
    DEVELOPMENT_CLEARANCE_M,
    DYNAMIC_MARGIN_GATE_RAD,
    FORBIDDEN_FORCE_GATE_N,
    HOLD_STEPS,
    JOINT_RATE_LIMIT_RAD_S,
    LEFT_ARM_JOINT_NAMES,
    MOVE_STEPS,
    PALM_BODY_NAMES,
    REFERENCE_PATH,
    REFERENCE_SHA256,
    RIGHT_ARM_JOINT_NAMES,
    ROOT_HEIGHT_RANGE_M,
    ROOT_TILT_GATE_DEG,
    SETTLE_STEPS,
    STATIC_MARGIN_GATE_RAD,
    STATIC_SEGMENT_SAMPLES,
    STATIC_STRAIGHT_SAMPLES,
    STAGE,
    TORQUE_RATIO_GATE,
    finite,
    minimum_jerk_fraction,
    mirror_audit,
    order_audit,
    path_joint_travel,
    segment_samples,
)
from g1_access_push.stage2.s2_03t_pelvis_contact_runtime import (  # noqa: E402
    CONTACT_EPS_N,
    BodyClearanceModel,
    ContactPairAuditor,
    EvidenceVideo,
    collider_audit,
    body_path_map,
    percentile,
    root_metrics,
    sha256,
    tolist,
)


ENV: Any = None
PAIR: ContactPairAuditor | None = None
VIDEOS: list[EvidenceVideo] = []


def load_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("TARGET_REFERENCE_MISSING")
    actual = sha256(path)
    if actual != REFERENCE_SHA256:
        raise RuntimeError(f"TARGET_REFERENCE_SHA_MISMATCH:{actual}")
    reference = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(reference, dict) or list(reference.get("arm_joint_names", ())) != list(ARM_JOINT_NAMES):
        raise RuntimeError("TARGET_REFERENCE_ARM_ORDER_INVALID")
    if not isinstance(reference.get("arm_ik_target"), torch.Tensor) or not finite(reference):
        raise RuntimeError("TARGET_REFERENCE_NONFINITE_OR_MISSING")
    return reference


def recurrent_reset_audit(env: Any) -> dict[str, Any]:
    lower = env.action_manager.get_term("lower_body_joint_pos")
    policy = getattr(lower, "_policy", None)
    values = {
        "hidden_state": getattr(policy, "hidden_state", None),
        "cell_state": getattr(policy, "cell_state", None),
        "previous_policy_action": getattr(lower, "_previous_policy_actions", None),
    }
    if not all(isinstance(value, torch.Tensor) for value in values.values()):
        return {"reset_verified": False, "reason": "RECURRENT_STATE_FIELDS_MISSING"}
    maxima = {f"{name}_abs_max": float(value.detach().abs().max()) for name, value in values.items()}
    ok = all(value <= 1.0e-8 for value in maxima.values())
    return {"reset_verified": ok, "reason": "ZEROED" if ok else "NONZERO_AFTER_RESET", **maxima}


def term_joint_ids(term: Any, count: int) -> list[int]:
    ids = getattr(term, "_joint_ids", None)
    if isinstance(ids, slice):
        return [int(value) for value in list(range(count))[ids]]
    return [] if ids is None else [int(value) for value in ids]


def runtime_audit(env: Any, cfg: Any, stage: Any) -> dict[str, Any]:
    robot = env.scene["robot"]
    arm_ids, arm_names = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
    left_ids, left_names = robot.find_joints(list(LEFT_ARM_JOINT_NAMES), preserve_order=True)
    right_ids, right_names = robot.find_joints(list(RIGHT_ARM_JOINT_NAMES), preserve_order=True)
    palm_ids, palm_names = robot.find_bodies(list(PALM_BODY_NAMES), preserve_order=True)
    wrist_ids, wrist_names = robot.find_bodies(["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True)
    if tuple(arm_names) != ARM_JOINT_NAMES or tuple(left_names) != LEFT_ARM_JOINT_NAMES or tuple(right_names) != RIGHT_ARM_JOINT_NAMES:
        raise RuntimeError("ARM_REFERENCE_ORDER_INVALID")
    frame_names = list(env.scene["hand_frames"].data.target_frame_names)
    if tuple(palm_names) != PALM_BODY_NAMES or not {"left_hand_palm", "right_hand_palm"}.issubset(frame_names):
        raise RuntimeError(f"END_EFFECTOR_FRAME_INVALID:{palm_names}:{frame_names}")
    names = list(env.action_manager.active_terms)
    dims = [int(value) for value in env.action_manager.action_term_dim]
    slices: dict[str, list[int]] = {}
    cursor = 0
    for name, dim in zip(names, dims, strict=True):
        slices[name] = [cursor, cursor + dim]
        cursor += dim
    overlap: dict[str, list[str]] = {}
    for name in names:
        term = env.action_manager.get_term(name)
        ids = term_joint_ids(term, robot.num_joints)
        overlap[name] = [robot.joint_names[index] for index in sorted(set(ids) & set(int(value) for value in arm_ids))]
    overridden = any(bool(value) for name, value in overlap.items() if name not in {"left_arm_joint_pos", "right_arm_joint_pos"})
    lower = env.action_manager.get_term("lower_body_joint_pos")
    checkpoint = Path(str(getattr(lower.cfg, "policy_path", "")))
    checkpoint_sha = sha256(checkpoint) if checkpoint.is_file() else None
    expected_checkpoint = "a0151975a757a33f0f5ed236d5616e2a98643abe2b36e50ac9ad01b6dd1f6d7e"
    if checkpoint_sha != expected_checkpoint:
        raise RuntimeError(f"LOWER_CHECKPOINT_SHA_MISMATCH:{checkpoint_sha}")
    paths = body_path_map(stage)
    limits = robot.data.joint_pos_limits[0, arm_ids].detach().cpu().tolist()
    effort = robot.data.joint_effort_limits[0, arm_ids].detach().cpu().tolist()
    return {
        "asset": {"usd_path": str(getattr(cfg.scene.robot.spawn, "usd_path", "METRIC_MISSING")), "robot_prim_path": "/World/envs/env_0/Robot"},
        "body_names": list(robot.body_names), "body_paths": paths,
        "wrist_body_names": list(wrist_names), "wrist_body_ids": [int(value) for value in wrist_ids],
        "end_effector_body_names": list(palm_names), "end_effector_body_ids": [int(value) for value in palm_ids],
        "palm_frame_names": frame_names,
        "arm_joint_names": list(arm_names), "arm_joint_ids": [int(value) for value in arm_ids],
        "left_arm_joint_names": list(left_names), "left_arm_joint_ids": [int(value) for value in left_ids],
        "right_arm_joint_names": list(right_names), "right_arm_joint_ids": [int(value) for value in right_ids],
        "arm_joint_limits_rad": limits, "arm_effort_limits": effort,
        "action_terms": {"names": names, "dimensions": dims, "slices": slices, "arm_overlap_by_term": overlap,
                         "lower_body_joint_names": list(getattr(lower, "_joint_names", ())),
                         "lower_body_checkpoint_path": str(checkpoint), "lower_body_checkpoint_sha256": checkpoint_sha,
                         "lower_body_command": [0.0, 0.0, 0.0, 0.7],
                         "upper_body_target_source": ["left_arm_joint_pos", "right_arm_joint_pos"],
                         "final_actuator_target_api": "JointPositionAction -> robot.set_joint_position_target"},
        "upper_body_target_overridden": overridden,
        "hand_visual_type": "Unitree G1 Dex3-1 hand (runtime USD audit)", "hardware_hand_match": "UNKNOWN",
        "no_box": True, "no_actor": True, "no_planner": True, "no_ppo": True, "no_base_command": True,
    }


def set_camera_views(env: Any) -> None:
    env.scene["safe_chest_front_camera"].set_world_poses_from_view(
        torch.tensor([[2.20, -2.80, 1.35]], device=env.device), torch.tensor([[0.0, 0.0, 0.84]], device=env.device))
    env.scene["safe_chest_side_camera"].set_world_poses_from_view(
        torch.tensor([[2.80, 0.05, 1.12]], device=env.device), torch.tensor([[0.0, 0.0, 0.86]], device=env.device))


def action_layout(env: Any) -> dict[str, slice]:
    names = list(env.action_manager.active_terms)
    dims = [int(value) for value in env.action_manager.action_term_dim]
    result: dict[str, slice] = {}
    cursor = 0
    for name, dim in zip(names, dims, strict=True):
        result[name] = slice(cursor, cursor + dim)
        cursor += dim
    required = {"left_arm_joint_pos", "right_arm_joint_pos", "waist_joint_pos", "finger_joint_pos", "lower_body_joint_pos"}
    if not required.issubset(result):
        raise RuntimeError(f"ACTION_LAYOUT_INVALID:{names}")
    result["__total__"] = slice(0, cursor)
    return result


def set_arm_q(env: Any, arm_ids: list[int], q: torch.Tensor) -> None:
    robot = env.scene["robot"]
    position, velocity = robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel)
    position[:, arm_ids] = q.reshape(1, -1)
    robot.write_joint_state_to_sim(position, velocity)
    env.sim.forward()
    try:
        env.scene.update(0.0)
    except TypeError:
        env.scene.update(env.sim.dt)


def forbidden_contact(env: Any) -> tuple[bool | None, float | None, list[str]]:
    sensor = env.scene["contact_forces"]
    values, names = sensor.data.net_forces_w, list(sensor.body_names)
    if values is None:
        return None, None, []
    norms = torch.linalg.vector_norm(values[0], dim=-1)
    if not bool(torch.isfinite(norms).all()) or len(names) != int(norms.numel()):
        return None, None, names
    bad: list[str] = []
    maximum = 0.0
    for name, force in zip(names, norms, strict=True):
        value = float(force)
        if "ankle" in name or "foot" in name:
            continue
        maximum = max(maximum, value)
        if value > FORBIDDEN_FORCE_GATE_N:
            bad.append(name)
    return bool(bad), maximum, bad


def pose_error(env: Any, body_ids: list[int], target_pos: torch.Tensor, target_quat: torch.Tensor) -> dict[str, Any]:
    robot = env.scene["robot"]
    count = len(body_ids)
    pos, quat = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w.unsqueeze(1).expand(-1, count, -1),
        robot.data.root_link_quat_w.unsqueeze(1).expand(-1, count, -1),
        robot.data.body_pos_w[:, body_ids], robot.data.body_quat_w[:, body_ids],
    )
    error = math_utils.compute_pose_error(pos[0], quat[0], target_pos, target_quat, rot_error_type="axis_angle")
    return {"pos": pos[0], "quat": quat[0], "error": error}


def make_record(env: Any, runtime: dict[str, Any], phase: str, segment: str, frame: int,
                desired_q: torch.Tensor, target_q: torch.Tensor, target_pos: torch.Tensor,
                target_quat: torch.Tensor, baseline_root: torch.Tensor, hold_timer_s: float,
                previous_target: torch.Tensor | None, clearance: BodyClearanceModel,
                reset_audit: dict[str, Any], waypoint_symbols: str) -> tuple[dict[str, Any], torch.Tensor]:
    robot = env.scene["robot"]
    arm_ids = [int(value) for value in runtime["arm_joint_ids"]]
    body_ids = [int(value) for value in runtime["end_effector_body_ids"]]
    actual_q = robot.data.joint_pos[0, arm_ids].detach().clone()
    actuator_target = robot.data.joint_pos_target[0, arm_ids].detach().clone()
    velocity = robot.data.joint_vel[0, arm_ids].detach().clone()
    limits = robot.data.joint_pos_limits[0, arm_ids]
    margins = torch.minimum(actual_q - limits[:, 0], limits[:, 1] - actual_q)
    target_margins = torch.minimum(actuator_target - limits[:, 0], limits[:, 1] - actuator_target)
    effort = robot.data.joint_effort_limits[0, arm_ids].abs().clamp_min(1.0e-6)
    torque = robot.data.applied_torque[0, arm_ids] if hasattr(robot.data, "applied_torque") else torch.zeros_like(actual_q)
    torque_ratio = float((torque.abs() / effort).max())
    poses = pose_error(env, body_ids, target_pos, target_quat)
    root_pos, root_quat = robot.data.root_link_pos_w[0], robot.data.root_link_quat_w[0]
    roll, pitch, yaw, tilt = root_metrics(root_quat)
    bad, force_max, bad_bodies = forbidden_contact(env)
    pair = PAIR.sample(frame, phase, actual_q, force_all=220 <= frame <= 246) if PAIR is not None else {"pair_force_norm_n": None, "closest_pair": "UNKNOWN", "best": None}
    left_clearance, right_clearance = clearance.clearance("left"), clearance.clearance("right")
    q_error = (actual_q - desired_q).abs()
    action_rate = 0.0 if previous_target is None else float((actuator_target - previous_target).abs().max() / CONTROL_DT_S)
    finite_state = bool(torch.isfinite(actual_q).all() and torch.isfinite(actuator_target).all() and torch.isfinite(velocity).all()
                        and torch.isfinite(poses["pos"]).all() and torch.isfinite(poses["quat"]).all()
                        and torch.isfinite(root_pos).all() and torch.isfinite(root_quat).all())
    record = {
        "frame": int(frame), "time_s": frame * CONTROL_DT_S, "phase": phase, "segment": segment,
        "finite": finite_state, "desired_upper_body_joints_rad": tolist(desired_q),
        "target_upper_body_joints_rad": tolist(target_q), "actual_upper_body_joints_rad": tolist(actual_q),
        "final_actuator_target_rad": tolist(actuator_target), "upper_joint_velocity_rad_s": tolist(velocity),
        "upper_joint_velocity_max_rad_s": float(velocity.abs().max()),
        "left_joint_tracking_max_error_rad": float(q_error[0::2].max()), "right_joint_tracking_max_error_rad": float(q_error[1::2].max()),
        "action_rate_max_rad_s": action_rate, "action_rate_source": "absolute JointPositionAction target delta / control_dt",
        "arm_torque_ratio_max": torque_ratio, "minimum_joint_limit_margin_rad": float(margins.min()),
        "minimum_target_joint_limit_margin_rad": float(target_margins.min()),
        "left_palm_position_m": tolist(poses["pos"][0]), "right_palm_position_m": tolist(poses["pos"][1]),
        "left_palm_quaternion_wxyz": tolist(poses["quat"][0]), "right_palm_quaternion_wxyz": tolist(poses["quat"][1]),
        "left_position_error_m": float(torch.linalg.vector_norm(poses["error"][0][0])),
        "right_position_error_m": float(torch.linalg.vector_norm(poses["error"][0][1])),
        "left_position_error_xyz_m": tolist(poses["error"][0][0]), "right_position_error_xyz_m": tolist(poses["error"][0][1]),
        "left_orientation_error_deg": math.degrees(float(torch.linalg.vector_norm(poses["error"][1][0]))),
        "right_orientation_error_deg": math.degrees(float(torch.linalg.vector_norm(poses["error"][1][1]))),
        "root_position_world_m": tolist(root_pos), "root_height_m": float(root_pos[2]),
        "root_roll_deg": math.degrees(roll), "root_pitch_deg": math.degrees(pitch), "root_yaw_deg": math.degrees(yaw),
        "root_tilt_deg": math.degrees(tilt), "root_displacement_xyz_m": float(torch.linalg.vector_norm(root_pos - baseline_root)),
        "forbidden_collision": bad, "forbidden_contact_force_max_n": force_max, "forbidden_contact_bodies": bad_bodies,
        "pair_force_norm_n": pair.get("pair_force_norm_n"), "closest_pair": pair.get("closest_pair"), "pair_contact_count": pair.get("contact_count", 0),
        "pair_best": pair.get("best"), "left_pelvis_clearance_m": left_clearance.get("clearance_m"),
        "right_pelvis_clearance_m": right_clearance.get("clearance_m"), "clearance_pair": left_clearance.get("pair"),
        "hold_timer_s": hold_timer_s, "recurrent_reset": reset_audit, "waypoint_symbols": waypoint_symbols,
        "upper_body_target_overridden": bool(runtime["upper_body_target_overridden"]),
        "action_source": {"left_arm": "left_arm_joint_pos:JointPositionAction", "right_arm": "right_arm_joint_pos:JointPositionAction",
                          "lower_body": "lower_body_joint_pos:frozen_recurrent_student", "base": "fixed_zero_command"},
    }
    return record, actuator_target


def detect_oscillation(records: list[dict[str, Any]]) -> bool:
    selected = [record for record in records if record.get("phase") == "HOLD"]
    if len(selected) < 12:
        return False
    points = np.asarray([[*record["left_palm_position_m"], *record["right_palm_position_m"]] for record in selected], dtype=float)
    velocity = np.diff(points, axis=0) / CONTROL_DT_S
    for axis in range(points.shape[1]):
        values = velocity[:, axis]
        values = values[np.abs(values) > 1.0e-4]
        reversals = int(np.sum(np.sign(values[1:]) * np.sign(values[:-1]) < 0)) if len(values) > 1 else 0
        if reversals >= 8 and float(points[:, axis].max() - points[:, axis].min()) > 0.003:
            return True
    return False


def run_episode(env: Any, runtime: dict[str, Any], clearance: BodyClearanceModel, *, label: str,
                waypoints: list[torch.Tensor], active: tuple[bool, bool], target_q: torch.Tensor,
                target_pos: torch.Tensor, target_quat: torch.Tensor, video: EvidenceVideo | None = None,
                stop_on_safety: bool = True) -> dict[str, Any]:
    observation, _ = env.reset(seed=42)
    if not finite(observation):
        raise RuntimeError(f"OBSERVATION_NONFINITE_AFTER_RESET:{label}")
    reset_audit = recurrent_reset_audit(env)
    if not reset_audit["reset_verified"]:
        raise RuntimeError(f"RECURRENT_RESET_INVALID:{label}:{reset_audit}")
    robot = env.scene["robot"]
    arm_ids = [int(value) for value in runtime["arm_joint_ids"]]
    waist_ids, _ = robot.find_joints(["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"], preserve_order=True)
    finger_names = [name for name in robot.joint_names if "hand_" in name]
    finger_ids, _ = robot.find_joints(finger_names, preserve_order=True)
    layout = action_layout(env)
    q0 = robot.data.joint_pos[0, arm_ids].detach().clone()
    q0_full = q0.clone()
    target_full = target_q.detach().clone()
    for index in range(14):
        if (index % 2 == 0 and not active[0]) or (index % 2 == 1 and not active[1]):
            target_full[index] = q0[index]
    path = [waypoints[0].detach().clone() if waypoints else q0_full]
    path[0] = q0_full
    for point in waypoints[1:]:
        adjusted = point.detach().clone()
        for index in range(14):
            if (index % 2 == 0 and not active[0]) or (index % 2 == 1 and not active[1]):
                adjusted[index] = q0[index]
        path.append(adjusted)
    if len(path) == 1 and label != "DEFAULT_ARMS_STAND_BASELINE":
        path.append(target_full)
    actions = torch.zeros((1, int(layout["__total__"].stop)), dtype=torch.float32, device=env.device)
    lower_command = torch.tensor([0.0, 0.0, 0.0, 0.7], dtype=torch.float32, device=env.device)
    waist_target = robot.data.default_joint_pos[0, waist_ids].detach().clone()
    finger_target = robot.data.default_joint_pos[0, finger_ids].detach().clone()
    baseline_root = robot.data.root_link_pos_w[0].detach().clone()
    previous_target: torch.Tensor | None = None
    previous_command = q0.clone()
    records: list[dict[str, Any]] = []
    early_done, safety_trigger, clip_count = False, None, 0
    frame = 0

    def send(phase: str, segment: str, command: torch.Tensor, hold_timer: float) -> None:
        nonlocal previous_target, previous_command, early_done, safety_trigger, clip_count, frame
        max_delta = JOINT_RATE_LIMIT_RAD_S * CONTROL_DT_S
        raw = command.detach().clone()
        delta = raw - previous_command
        command = previous_command + torch.clamp(delta, -max_delta, max_delta)
        if bool((command - raw).abs().max() > 1.0e-8):
            clip_count += 1
        actions.zero_()
        actions[0, layout["left_arm_joint_pos"]] = command[0::2]
        actions[0, layout["right_arm_joint_pos"]] = command[1::2]
        actions[0, layout["waist_joint_pos"]] = waist_target
        actions[0, layout["finger_joint_pos"]] = finger_target
        actions[0, layout["lower_body_joint_pos"]] = lower_command
        observation, _ = env.step(actions)
        if not finite(observation):
            raise RuntimeError(f"NONFINITE_RUNTIME_STATE:{label}:{frame}")
        record, previous_target = make_record(env, runtime, phase, segment, frame, command, target_full, target_pos, target_quat,
                                              baseline_root, hold_timer, previous_target, clearance, reset_audit, "q_clear/q_lift/q_chest")
        records.append(record)
        previous_command = command.detach().clone()
        trigger = None
        if not record["finite"]:
            trigger = "METRIC_OR_STATE_NONFINITE"
        elif record["forbidden_collision"] is True:
            trigger = "FORBIDDEN_COLLISION"
        elif record["pair_force_norm_n"] is not None and float(record["pair_force_norm_n"]) > FORBIDDEN_FORCE_GATE_N:
            trigger = "PELVIS_CONTACT_FORCE"
        elif not (ROOT_HEIGHT_RANGE_M[0] <= record["root_height_m"] <= ROOT_HEIGHT_RANGE_M[1]):
            trigger = "ROOT_HEIGHT"
        elif record["root_tilt_deg"] > ROOT_TILT_GATE_DEG:
            trigger = "ROOT_TILT"
        elif record["minimum_joint_limit_margin_rad"] < DYNAMIC_MARGIN_GATE_RAD:
            trigger = "ACTUAL_JOINT_MARGIN"
        elif record["minimum_target_joint_limit_margin_rad"] < DYNAMIC_MARGIN_GATE_RAD:
            trigger = "TARGET_JOINT_MARGIN"
        elif record["arm_torque_ratio_max"] > TORQUE_RATIO_GATE:
            trigger = "ARM_TORQUE_RATIO"
        if trigger is not None and stop_on_safety:
            safety_trigger, early_done = trigger, True
        if video is not None:
            key = "first_contact" if record["pair_force_norm_n"] and float(record["pair_force_norm_n"]) > 1.0e-6 and not any(r.get("pair_force_norm_n", 0.0) for r in records[:-1]) else None
            video.capture(record, force=key is not None, keyframe=key)
        if frame % 25 == 0 or frame == 1:
            print(f"PHASE={phase} step={frame}", flush=True)

    for index in range(1, SETTLE_STEPS + 1):
        frame = index
        send("BASELINE", "settle", q0, 0.0)
        if early_done:
            break
    if not early_done:
        for segment_index in range(len(path) - 1):
            start, goal = path[segment_index], path[segment_index + 1]
            for index in range(1, MOVE_STEPS + 1):
                frame += 1
                fraction = minimum_jerk_fraction(index, MOVE_STEPS)
                send("MOVE", f"segment_{segment_index + 1}", start + fraction * (goal - start), 0.0)
                if early_done:
                    break
            if early_done:
                break
    if not early_done:
        for index in range(1, HOLD_STEPS + 1):
            frame += 1
            send("HOLD", "hold", target_full, index * CONTROL_DT_S)
            if video is not None and index == 1:
                video.capture(records[-1], force=True, keyframe="precontact")
            if early_done:
                break
    oscillation = detect_oscillation(records)
    for record in records:
        record["oscillation_detected"] = oscillation
    if video is not None and records:
        video.capture(records[-1], force=True, keyframe="terminal")
    hold_records = [record for record in records if record["phase"] == "HOLD"]
    safety_ok = bool(records) and all(
        record["finite"] and record["forbidden_collision"] is False
        and record["pair_force_norm_n"] is not None and float(record["pair_force_norm_n"]) <= FORBIDDEN_FORCE_GATE_N
        and ROOT_HEIGHT_RANGE_M[0] <= record["root_height_m"] <= ROOT_HEIGHT_RANGE_M[1]
        and record["root_tilt_deg"] <= ROOT_TILT_GATE_DEG
        and record["minimum_joint_limit_margin_rad"] >= DYNAMIC_MARGIN_GATE_RAD
        and record["minimum_target_joint_limit_margin_rad"] >= DYNAMIC_MARGIN_GATE_RAD
        and record["arm_torque_ratio_max"] <= TORQUE_RATIO_GATE for record in records)
    return {
        "label": label, "active_arms": {"left": active[0], "right": active[1]},
        "observed_control_frames": len(records),
        "expected_control_frames": SETTLE_STEPS + (len(path) - 1) * MOVE_STEPS + HOLD_STEPS,
        "hold_frames": len(hold_records), "hold_seconds": len(hold_records) * CONTROL_DT_S,
        "early_done": early_done, "safety_trigger": safety_trigger, "safety_ok": safety_ok,
        "recurrent_reset": reset_audit, "oscillation_detected": oscillation, "command_clip_count": clip_count,
        "left_position_error_p95_m": percentile(record["left_position_error_m"] for record in records),
        "right_position_error_p95_m": percentile(record["right_position_error_m"] for record in records),
        "left_position_error_max_m": max((record["left_position_error_m"] for record in records), default=None),
        "right_position_error_max_m": max((record["right_position_error_m"] for record in records), default=None),
        "root_tilt_max_deg": max((record["root_tilt_deg"] for record in records), default=None),
        "root_height_min_m": min((record["root_height_m"] for record in records), default=None),
        "root_height_max_m": max((record["root_height_m"] for record in records), default=None),
        "max_torque_ratio": max((record["arm_torque_ratio_max"] for record in records), default=None),
        "records": records,
    }


def static_state(env: Any, runtime: dict[str, Any], q: torch.Tensor, clearance: BodyClearanceModel, phase: str) -> dict[str, Any]:
    set_arm_q(env, [int(value) for value in runtime["arm_joint_ids"]], q)
    robot = env.scene["robot"]
    limits = robot.data.joint_pos_limits[0, runtime["arm_joint_ids"]]
    margin = float(torch.minimum(q - limits[:, 0], limits[:, 1] - q).min())
    bad, force, bodies = forbidden_contact(env)
    left, right = clearance.clearance("left"), clearance.clearance("right")
    pair = PAIR.sample(0, phase, q, force_all=False, record_history=False) if PAIR is not None else {"pair_force_norm_n": None, "closest_pair": "UNKNOWN"}
    finite_state = bool(torch.isfinite(q).all() and math.isfinite(margin) and force is not None)
    collision = bool((force or 0.0) > CONTACT_EPS_N or (pair.get("pair_force_norm_n") or 0.0) > CONTACT_EPS_N or (left.get("clearance_m") is not None and left["clearance_m"] < 0.0))
    return {"q_rad": tolist(q), "minimum_joint_margin_rad": margin, "forbidden_collision": bad,
            "forbidden_force_max_n": force, "forbidden_bodies": bodies, "pair_force_norm_n": pair.get("pair_force_norm_n"),
            "closest_pair": pair.get("closest_pair"), "left_clearance_m": left.get("clearance_m"),
            "right_clearance_m": right.get("clearance_m"), "left_clearance_pair": left.get("pair"),
            "finite": finite_state, "collision": collision, "phase": phase}


def static_segment(env: Any, runtime: dict[str, Any], start: torch.Tensor, goal: torch.Tensor,
                   samples: int, clearance: BodyClearanceModel, label: str) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    first, last, max_penetration = None, None, 0.0
    minimum_clearance: float | None = None
    for index, q_values in enumerate(segment_samples(start.tolist(), goal.tolist(), samples)):
        q = torch.tensor(q_values, dtype=torch.float32, device=env.device)
        value = static_state(env, runtime, q, clearance, "STATIC")
        value["sample_index"], value["s"] = index, index / float(samples - 1)
        points.append(value)
        current = value.get("left_clearance_m")
        if isinstance(current, (int, float)):
            minimum_clearance = current if minimum_clearance is None else min(minimum_clearance, current)
            max_penetration = max(max_penetration, -float(current))
        if value["collision"] and first is None:
            first = value["s"]
        if value["collision"]:
            last = value["s"]
        if index % 50 == 0 or index == samples - 1:
            print(f"STATIC_SEGMENT={label} sample={index}/{samples - 1}", flush=True)
    return {"label": label, "samples": samples, "first_colliding_s": first, "last_colliding_s": last,
            "max_penetration_m": max_penetration, "minimum_left_clearance_m": minimum_clearance,
            "collision_free": first is None and last is None, "records": points}


def make_waypoint_candidates(q0: torch.Tensor, q_chest: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # Deterministic, bounded templates: left shoulder/arm first; the endpoint
    # is copied exactly and is never searched or altered.
    candidates: list[tuple[torch.Tensor, torch.Tensor]] = []
    templates = ((0.45, -0.20, 0.10), (0.60, -0.25, 0.15), (0.35, -0.35, 0.20), (0.75, -0.15, 0.05),
                 (0.55, 0.10, 0.10), (0.40, 0.20, -0.10), (0.80, 0.00, 0.15), (0.65, -0.40, 0.25))
    for roll, yaw, pitch in templates:
        clear, lift = q0.clone(), q0.clone()
        clear[0], clear[2], clear[4], clear[6], clear[8], clear[10], clear[12] = q0[0] + pitch, roll, yaw, q0[6] + 0.10, q0[8], q0[10], q0[12]
        lift[0], lift[2], lift[4], lift[6], lift[8], lift[10], lift[12] = q_chest[0] * 0.65, roll * 0.65 + q_chest[2] * 0.35, yaw * 0.65 + q_chest[4] * 0.35, q_chest[6] * 0.60, q_chest[8] * 0.45, q_chest[10] * 0.35, q_chest[12] * 0.35
        # Keep the right arm at default for the left-first search.
        clear[1::2], lift[1::2] = q0[1::2], q0[1::2]
        candidates.append((clear, lift))
    return candidates


def update_waypoint_yaml(selected: dict[str, Any] | None) -> None:
    import yaml
    path = Path("configs/stage2/s2_03t_safe_chest_waypoint_path.yaml")
    if not path.is_file():
        return
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return
    search = value.setdefault("waypoint_search", {})
    choice = search.setdefault("selected", {})
    if selected is None:
        choice.update({"status": "NOT_SELECTED", "q_clear_arm_rad": None, "q_lift_arm_rad": None,
                       "minimum_pelvis_clearance_m": None, "static_collision_free": False, "selection_score": None})
        return
    value["status"] = "STATIC_COLLISION_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED"
    choice.update({"status": "STATIC_COLLISION_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED",
                   "q_clear_arm_rad": selected["q_clear_rad"], "q_lift_arm_rad": selected["q_lift_rad"],
                   "minimum_pelvis_clearance_m": selected["minimum_pelvis_clearance_m"], "static_collision_free": True,
                   "selection_score": selected["selection_score"]})
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
