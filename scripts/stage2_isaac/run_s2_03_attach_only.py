#!/usr/bin/env python3
"""Run S2-03 infrastructure preflight or the one attach-only formal episode."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

from g1_access_push.stage2.s2_01_process import validate_controller_checkpoint, write_implementation_exception

bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument("--run-root", type=Path, required=True)
bootstrap_args, _ = bootstrap.parse_known_args()
RUN = bootstrap_args.run_root.resolve()
RUNTIME_STATE = {"environment_created": False, "observed_frames": 0}
RUNTIME_ENV = None

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--resolved-config", type=Path, required=True)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--preflight-only", action="store_true")
mode.add_argument("--formal", action="store_true")
parser.add_argument("--preflight-steps", type=int, default=3)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import carb  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

from agile.rl_env.assets.robots.unitree_g1 import G1_W_HANDS_AGILE_ACTION_SCALE  # noqa: E402
from g1_access_push.sim.stage1.no_box_env_cfg import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES  # noqa: E402
from g1_access_push.sim.stage2.s2_03_attach_env import (  # noqa: E402
    BOX_FILTER_EXPRESSIONS,
    LEFT_PALM_SENSOR_PRIM_PATH,
    RIGHT_PALM_SENSOR_PRIM_PATH,
    build_s2_03_env_cfg,
)
from g1_access_push.stage2.attach_fsm import AttachState, next_state  # noqa: E402
from g1_access_push.stage2.s2_02_contract import object_local_targets  # noqa: E402
from g1_access_push.stage2.s2_03_contract import (  # noqa: E402
    advance_rate_limited,
    load_config,
    sha256_file,
    write_json,
)

BOX_PRIM_PATH = "/World/envs/env_0/Box"
ROBOT_PRIM_PATH = "/World/envs/env_0/Robot"


def process_count() -> int:
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "run_s2_03_attach_only.py" in command:
            count += 1
    return count


def term_slice(names: list[str], dims: list[int], name: str) -> slice:
    index = names.index(name)
    start = sum(dims[:index])
    return slice(start, start + dims[index])


def clamp_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value * torch.clamp(limit / norm.clamp_min(1.0e-9), max=1.0)


def quaternion_error_deg(actual: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(actual * desired, dim=-1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


def quaternion_rpy_tilt(q: torch.Tensor) -> tuple[float, float, float, float]:
    w, x, y, z = [float(item) for item in q]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    tilt = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))
    return roll, pitch, yaw, tilt


def wrapped_abs(value: float, reference: float) -> float:
    return abs(math.atan2(math.sin(value - reference), math.cos(value - reference)))


def set_runtime_mass_properties(box, object_cfg: dict) -> dict:
    view = getattr(box, "root_physx_view", None) or getattr(box, "root_view", None)
    if view is None:
        raise RuntimeError("RUNTIME_MASS_VIEW_UNAVAILABLE")
    masses = view.get_masses().clone()
    coms = view.get_coms().clone()
    inertias = view.get_inertias().clone()
    masses[...] = float(object_cfg["mass_kg"])
    coms[..., :3] = torch.tensor(object_cfg["center_of_mass_local_xyz_m"], device=coms.device, dtype=coms.dtype)
    coms[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=coms.device, dtype=coms.dtype)
    inertias.zero_()
    diagonal = object_cfg["diagonal_inertia_kg_m2"]
    for tensor_index, source_index in ((0, 0), (4, 1), (8, 2)):
        inertias[..., tensor_index] = float(diagonal[source_index])
    indices = torch.arange(int(view.count), dtype=torch.int32, device="cpu")
    view.set_masses(masses, indices)
    view.set_inertias(inertias, indices)
    view.set_coms(coms, indices)
    actual_mass = view.get_masses().clone()
    actual_com = view.get_coms().clone()
    actual_inertia = view.get_inertias().clone()
    first_com = actual_com.reshape(-1, actual_com.shape[-1])[0]
    first_inertia = actual_inertia.reshape(-1, actual_inertia.shape[-1])[0]
    checks = {
        "mass": abs(float(actual_mass.reshape(-1)[0]) - float(object_cfg["mass_kg"])) <= 1.0e-5,
        "com": max(abs(float(first_com[i]) - float(object_cfg["center_of_mass_local_xyz_m"][i])) for i in range(3)) <= 1.0e-5,
        "inertia": max(abs(float(first_inertia[i]) - float(diagonal[j])) for i, j in ((0, 0), (4, 1), (8, 2))) <= 1.0e-5,
        "principal_axes": max(abs(float(first_com[i]) - expected) for i, expected in zip(range(3, 7), (0.0, 0.0, 0.0, 1.0), strict=True)) <= 1.0e-5,
    }
    return {
        "runtime_view": type(view).__qualname__, "mass_kg": float(actual_mass.reshape(-1)[0]),
        "com_local_xyz_m": [float(value) for value in first_com[:3]],
        "principal_axes_xyzw": [float(value) for value in first_com[3:7]],
        "diagonal_inertia_kg_m2": [float(first_inertia[index]) for index in (0, 4, 8)],
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }


def overlap_query(box_pos: torch.Tensor, box_quat: torch.Tensor) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []

    def callback(hit) -> bool:
        collision = str(getattr(hit, "collision", ""))
        rigid_body = str(getattr(hit, "rigid_body", ""))
        if collision.startswith(ROBOT_PRIM_PATH) or rigid_body.startswith(ROBOT_PRIM_PATH):
            hits.append({"collision": collision, "rigid_body": rigid_body})
        return True

    get_physx_scene_query_interface().overlap_box(
        carb.Float3(0.6, 0.3, 0.6),
        carb.Float3(*[float(value) for value in box_pos]),
        carb.Float4(float(box_quat[1]), float(box_quat[2]), float(box_quat[3]), float(box_quat[0])),
        callback, False,
    )
    return hits


def place_box(box, center_x: float, center_y: float, center_z: float) -> None:
    state = box.data.default_root_state.clone()
    state[0, :3] = torch.tensor([center_x, center_y, center_z], device=state.device)
    state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device)
    state[0, 7:] = 0.0
    box.write_root_state_to_sim(state)


def target_pose_in_pelvis(robot, box_pos: torch.Tensor, box_quat: torch.Tensor, candidate: dict, desired_q_object: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local_positions = torch.tensor(object_local_targets(candidate), device=robot.device, dtype=torch.float32)
    target_world_pos, target_world_quat = math_utils.combine_frame_transforms(
        box_pos.expand(2, 3), box_quat.expand(2, 4), local_positions, desired_q_object.expand(2, 4),
    )
    return math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w[0].expand(2, 3), robot.data.root_link_quat_w[0].expand(2, 4),
        target_world_pos, target_world_quat,
    )


def audit_palm_sensor(stage, sensor, configured_path: str, expected_body_name: str) -> dict:
    force_matrix = sensor.data.force_matrix_w
    actual_paths = list(sensor.body_physx_view.prim_paths[: sensor.num_bodies])
    actual_path = actual_paths[0] if len(actual_paths) == 1 else ""
    prim = stage.GetPrimAtPath(actual_path)
    checks = {
        "initialized": sensor.is_initialized,
        "one_body": sensor.num_bodies == 1,
        "body_name": list(sensor.body_names) == [expected_body_name],
        "one_box_filter": int(sensor.contact_physx_view.filter_count) == 1,
        "force_matrix_shape": force_matrix is not None and list(force_matrix.shape) == [1, 1, 1, 3],
        "force_matrix_finite": force_matrix is not None and bool(torch.isfinite(force_matrix).all()),
        "contact_reporter": bool(prim.IsValid() and prim.HasAPI(PhysxSchema.PhysxContactReportAPI)),
        "rigid_body": bool(prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI)),
        "box_filter_config": list(sensor.cfg.filter_prim_paths_expr) == list(BOX_FILTER_EXPRESSIONS),
    }
    return {
        "configured_prim_path": configured_path, "resolved_prim_expression": sensor.cfg.prim_path,
        "actual_body_paths": actual_paths, "body_names": list(sensor.body_names),
        "filter_count": int(sensor.contact_physx_view.filter_count),
        "force_matrix_shape": None if force_matrix is None else list(force_matrix.shape),
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> None:
    global RUNTIME_ENV
    RUN.mkdir(parents=True, exist_ok=True)
    run_mode = "preflight" if args.preflight_only else "formal"
    if args.preflight_only and not 2 <= args.preflight_steps <= 5:
        raise ValueError("preflight steps must be within [2, 5]")
    status = {
        "status": "RUNNING", "primary_reason": None, "mode": run_mode,
        "environment_created": False, "observed_frames": 0,
        "multiple_isaac_processes": process_count() > 1,
    }
    write_json(RUN / "runner_status.json", status)
    if status["multiple_isaac_processes"]:
        raise RuntimeError("MULTIPLE_ISAAC_PROCESSES")
    config = load_config(args.config)
    resolved = json.loads(args.resolved_config.read_text(encoding="utf-8"))
    if resolved.get("config_sha256") != sha256_file(args.config) or not resolved.get("runnable"):
        raise RuntimeError("CONFIG_SHA_MISMATCH")
    s2_02 = Path(config["certification"]["s2_02_formal_result_path"])
    if not s2_02.is_file() or json.loads(s2_02.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("S2_02_PASS_EVIDENCE_MISSING")

    cfg = build_s2_03_env_cfg()
    cfg.seed = 42
    cfg.sim.device = args.device
    env = ManagerBasedEnv(cfg=cfg)
    RUNTIME_ENV = env
    RUNTIME_STATE["environment_created"] = True
    status["environment_created"] = True
    write_json(RUN / "runner_status.json", status)
    env.reset(seed=cfg.seed)
    print("PHASE=RESET", flush=True)

    stage = get_current_stage()
    robot = env.scene["robot"]
    box = env.scene["box"]
    palms = env.scene["hand_frames"]
    left_sensor = env.scene["left_palm_box_contact"]
    right_sensor = env.scene["right_palm_box_contact"]
    camera = env.scene["audit_camera"]
    lower = env.action_manager.get_term("lower_body_joint_pos")
    baseline_root_pos = robot.data.root_link_pos_w[0].clone()
    place_box(
        box,
        float(baseline_root_pos[0]) + float(config["precontact"]["base_to_box_center_distance_m"]),
        float(baseline_root_pos[1]), float(config["object"]["spawn_center_z_m"]),
    )

    checkpoint = Path(lower.cfg.policy_path).resolve()
    checkpoint_audit = validate_controller_checkpoint(
        checkpoint, sha256_file(checkpoint), config["certification"]["controller_checkpoint_sha256"],
        [config["certification"]["forbidden_checkpoint_sha256"]], [],
    )
    recurrent_reset = {
        "hidden_state_all_zero": bool(torch.count_nonzero(lower._policy.hidden_state).item() == 0),
        "cell_state_all_zero": bool(torch.count_nonzero(lower._policy.cell_state).item() == 0),
        "previous_policy_action_all_zero": bool(torch.count_nonzero(lower._previous_policy_actions).item() == 0),
        "last_policy_input_all_zero": bool(torch.count_nonzero(lower.last_policy_input).item() == 0),
    }
    if checkpoint_audit["status"] != "PASS" or not all(recurrent_reset.values()):
        raise RuntimeError("CHECKPOINT_CONTRACT_FAILED")

    names = list(env.action_manager.active_terms)
    dims = [int(value) for value in env.action_manager.action_term_dim]
    action_contract = dict(zip(names, dims, strict=True))
    expected_action_contract = {"left_hand_pose": 6, "right_hand_pose": 6, "waist_joint_pos": 3, "lower_body_joint_pos": 4}
    if action_contract != expected_action_contract:
        raise RuntimeError(f"ACTION_CONTRACT_MISMATCH:{action_contract}")
    left_slice = term_slice(names, dims, "left_hand_pose")
    right_slice = term_slice(names, dims, "right_hand_pose")
    waist_slice = term_slice(names, dims, "waist_joint_pos")
    lower_slice = term_slice(names, dims, "lower_body_joint_pos")
    actions = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    lower_command = torch.tensor(config["controller"]["lower_body_command"], device=env.device)

    mass_audit = set_runtime_mass_properties(box, config["object"])
    write_json(RUN / "runtime_mass_properties_audit.json", mass_audit)
    if mass_audit["status"] != "PASS":
        raise RuntimeError("RUNTIME_MASS_PROPERTIES_FAILED")
    left_audit = audit_palm_sensor(stage, left_sensor, LEFT_PALM_SENSOR_PRIM_PATH, "left_hand_palm_link")
    right_audit = audit_palm_sensor(stage, right_sensor, RIGHT_PALM_SENSOR_PRIM_PATH, "right_hand_palm_link")
    sensor_audit = {
        "schema_version": 1, "left": left_audit, "right": right_audit,
        "separate_sensor_instances": left_sensor is not right_sensor,
        "status": "PASS" if left_audit["status"] == right_audit["status"] == "PASS" and left_sensor is not right_sensor else "FAIL",
    }
    write_json(RUN / "palm_contact_sensor_audit.json", sensor_audit)
    if sensor_audit["status"] != "PASS":
        raise RuntimeError("PALM_CONTACT_SENSOR_INITIALIZATION_FAILED")

    frame_names = list(palms.data.target_frame_names)
    left_frame = frame_names.index("left_hand_palm")
    right_frame = frame_names.index("right_hand_palm")
    palm_ids, palm_body_names = robot.find_bodies(["left_hand_palm_link", "right_hand_palm_link"], preserve_order=True)
    if (left_frame, right_frame) != (0, 1) or palm_body_names != ["left_hand_palm_link", "right_hand_palm_link"]:
        raise RuntimeError("PALM_FRAME_OR_BODY_ORDER_MISMATCH")
    allowed_palm_paths = set(left_audit["actual_body_paths"] + right_audit["actual_body_paths"])
    arm_ids, arm_names = robot.find_joints(LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES)
    joint_limits = robot.data.joint_pos_limits[0, arm_ids].clone()
    effort_limits = robot.data.joint_effort_limits[0, arm_ids].abs().clamp_min(1.0e-6)
    write_json(RUN / "runtime_geometry_audit.json", {
        "schema_version": 1, "palm_frame_names": frame_names, "palm_body_names": palm_body_names,
        "palm_body_ids": palm_ids, "allowed_palm_runtime_paths": sorted(allowed_palm_paths),
        "arm_joint_names": arm_names, "arm_joint_limits_rad": joint_limits.detach().cpu().tolist(),
        "robot_body_names": list(robot.body_names), "box_prim_path": BOX_PRIM_PATH,
        "status": "PASS" if len(arm_names) == 14 and len(allowed_palm_paths) == 2 else "FAIL",
    })
    write_json(RUN / "controller_contract_audit.json", {
        **checkpoint_audit, "recurrent_reset": recurrent_reset, "action_contract": action_contract,
        "lower_body_scale_source": "G1_W_HANDS_AGILE_ACTION_SCALE",
        "lower_body_scale_source_entry_count": len(G1_W_HANDS_AGILE_ACTION_SCALE),
        "resolved_lower_body_scale": [float(value) for value in lower._policy_output_scale[0]],
        "resolved_lower_body_offset": [float(value) for value in lower._policy_output_offset[0]],
        "lower_body_command": config["controller"]["lower_body_command"],
    })
    camera.set_world_poses_from_view(
        torch.tensor([[2.0, -2.0, 1.55]], device=env.device),
        torch.tensor([[0.45, 0.0, 0.70]], device=env.device),
    )

    maximum_position = float(config["controller"]["maximum_position_correction_m"])
    maximum_orientation = float(config["controller"]["maximum_orientation_correction_rad"])

    def command(desired_positions: torch.Tensor, desired_quaternions: torch.Tensor) -> None:
        current_positions = palms.data.target_pos_source[0]
        current_quaternions = palms.data.target_quat_source[0]
        position_error, orientation_error = math_utils.compute_pose_error(
            current_positions, current_quaternions, desired_positions, desired_quaternions,
            rot_error_type="axis_angle",
        )
        actions.zero_()
        actions[:, left_slice] = torch.cat((clamp_norm(position_error, maximum_position)[left_frame], clamp_norm(orientation_error, maximum_orientation)[left_frame]))
        actions[:, right_slice] = torch.cat((clamp_norm(position_error, maximum_position)[right_frame], clamp_norm(orientation_error, maximum_orientation)[right_frame]))
        actions[:, waist_slice] = 0.0
        actions[:, lower_slice] = lower_command
        if not bool(torch.isfinite(actions).all()):
            raise RuntimeError("NONFINITE")
        env.step(actions)

    if args.preflight_only:
        last_image = None
        with (RUN / "trace.jsonl").open("w", encoding="utf-8", buffering=1) as trace:
            for frame in range(args.preflight_steps):
                actions.zero_()
                actions[:, lower_slice] = lower_command
                env.step(actions)
                finite = bool(torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(lower.last_policy_input).all())
                trace.write(json.dumps({"frame": frame, "finite": finite, "mode": "preflight"}, sort_keys=True) + "\n")
                RUNTIME_STATE["observed_frames"] = frame + 1
                status["observed_frames"] = frame + 1
                last_image = Image.fromarray(camera.data.output["rgb"][0].detach().cpu().numpy().astype("uint8"))
                print(f"PHASE=PREFLIGHT step={frame + 1}/{args.preflight_steps}", flush=True)
        if last_image is not None:
            last_image.save(RUN / "preflight.png")
        checks = {
            "environment_created": True, "runtime_mass_properties": mass_audit["status"] == "PASS",
            "left_palm_sensor": left_audit["status"] == "PASS", "right_palm_sensor": right_audit["status"] == "PASS",
            "separate_palm_sensors": left_sensor is not right_sensor,
            "controller_contract": checkpoint_audit["status"] == "PASS" and all(recurrent_reset.values()),
            "runtime_geometry": len(allowed_palm_paths) == 2 and len(arm_names) == 14,
            "trace_writer": RUNTIME_STATE["observed_frames"] == args.preflight_steps,
            "camera_output": (RUN / "preflight.png").is_file(), "no_second_isaac": not status["multiple_isaac_processes"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        write_json(RUN / "preflight_result.json", {
            "schema_version": 1, "stage": "S2-03", "mode": "PREFLIGHT_ONLY",
            "status": "PASS" if not failed else "FAIL",
            "primary_reason": "ALL_PREFLIGHT_GATES_PASSED" if not failed else "PREFLIGHT_GATES_FAILED",
            "checks": checks, "failed_checks": failed, "steps": args.preflight_steps,
            "config_sha256": sha256_file(args.config), "resolved_config_sha256": sha256_file(args.resolved_config),
            "scientific_result_created": False,
        })
        status.update({"status": "COMPLETE" if not failed else "PREFLIGHT_FAILED", "primary_reason": None if not failed else "PREFLIGHT_GATES_FAILED"})
        write_json(RUN / "runner_status.json", status)
        return

    for step in range(100):
        actions.zero_()
        actions[:, lower_slice] = lower_command
        env.step(actions)
        if (step + 1) % 25 == 0:
            print(f"PHASE=warmup step={step + 1}", flush=True)

    baseline_positions = palms.data.target_pos_source[0].clone()
    baseline_quaternions = palms.data.target_quat_source[0].clone()
    root_reference = robot.data.root_link_pos_w[0].clone()
    box_fixed_pos = box.data.root_link_pos_w[0].clone()
    box_fixed_quat = box.data.root_link_quat_w[0].clone()
    box_attach_reference = box_fixed_pos.clone()
    _, _, box_yaw_reference, _ = quaternion_rpy_tilt(box_fixed_quat)
    desired_q_object = torch.tensor(config["precontact"]["desired_palm_quaternion_in_object_wxyz"], device=env.device, dtype=torch.float32)
    candidate_base = {
        key: config["precontact"][key]
        for key in ("contact_height_m", "tangential_separation_m", "base_to_box_center_distance_m", "precontact_gap_m", "palm_collision_support_offset_m")
    }

    fsm_state = AttachState.RESET
    frame_index = 0
    transitions = [{"frame": -1, "state": AttachState.RESET.value, "reason": None}]
    tracker = {
        "left_impulse": 0.0, "right_impulse": 0.0, "previous_left_force": 0.0, "previous_right_force": 0.0,
        "left_onset": None, "right_onset": None, "bilateral_onset": None,
        "left_losses": [], "right_losses": [], "previous_left_contact": False, "previous_right_contact": False,
        "last_image": None,
    }
    dt = float(config["controller"]["control_dt_s"])

    def transition(requested: AttachState, reason: str | None = None) -> None:
        nonlocal fsm_state
        fsm_state = next_state(fsm_state, requested, failure_reason=reason)
        transitions.append({"frame": frame_index - 1, "state": fsm_state.value, "reason": reason})
        print(f"PHASE={fsm_state.value} frame={frame_index} reason={reason}", flush=True)

    transition(AttachState.STAND_SETTLE)

    def step_record(trace, desired_positions: torch.Tensor, desired_quaternions: torch.Tensor, subphase: str, approach_displacement: float, *, failure_unload: bool = False) -> dict:
        nonlocal frame_index
        command(desired_positions, desired_quaternions)
        left_force = float(torch.linalg.vector_norm(left_sensor.data.force_matrix_w[0, 0, 0]).item())
        right_force = float(torch.linalg.vector_norm(right_sensor.data.force_matrix_w[0, 0, 0]).item())
        left_contact = left_force >= float(config["contact_gate"]["left_contact_force_threshold_n"])
        right_contact = right_force >= float(config["contact_gate"]["right_contact_force_threshold_n"])
        if left_contact and tracker["left_onset"] is None:
            tracker["left_onset"] = frame_index
        if right_contact and tracker["right_onset"] is None:
            tracker["right_onset"] = frame_index
        impulse_window = int(config["motion"]["contact_impulse_window_steps"])
        if tracker["left_onset"] is not None and frame_index - tracker["left_onset"] < impulse_window:
            tracker["left_impulse"] += left_force * dt
        if tracker["right_onset"] is not None and frame_index - tracker["right_onset"] < impulse_window:
            tracker["right_impulse"] += right_force * dt
        if left_contact and right_contact and tracker["bilateral_onset"] is None:
            tracker["bilateral_onset"] = frame_index
        if tracker["previous_left_contact"] and not left_contact:
            tracker["left_losses"].append(frame_index)
        if tracker["previous_right_contact"] and not right_contact:
            tracker["right_losses"].append(frame_index)
        force_rate = max(abs(left_force - tracker["previous_left_force"]), abs(right_force - tracker["previous_right_force"])) / dt
        tracker["previous_left_force"] = left_force
        tracker["previous_right_force"] = right_force
        tracker["previous_left_contact"] = left_contact
        tracker["previous_right_contact"] = right_contact

        actual_positions = palms.data.target_pos_source[0]
        actual_quaternions = palms.data.target_quat_source[0]
        palm_world_pos, palm_world_quat = math_utils.combine_frame_transforms(
            robot.data.root_link_pos_w[0].expand(2, 3), robot.data.root_link_quat_w[0].expand(2, 4),
            actual_positions, actual_quaternions,
        )
        palm_object_pos, _ = math_utils.subtract_frame_transforms(
            box.data.root_link_pos_w[0].expand(2, 3), box.data.root_link_quat_w[0].expand(2, 4),
            palm_world_pos, palm_world_quat,
        )
        hits = overlap_query(box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
        forbidden = sorted({
            hit["rigid_body"] or hit["collision"]
            for hit in hits if (hit["rigid_body"] or hit["collision"]) not in allowed_palm_paths
        })
        arm_pos = robot.data.joint_pos[0, arm_ids]
        lower_margin = arm_pos - joint_limits[:, 0]
        upper_margin = joint_limits[:, 1] - arm_pos
        root_roll, root_pitch, _, root_tilt = quaternion_rpy_tilt(robot.data.root_link_quat_w[0])
        _, _, box_yaw, _ = quaternion_rpy_tilt(box.data.root_link_quat_w[0])
        finite = bool(
            torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(robot.data.joint_pos).all()
            and torch.isfinite(robot.data.applied_torque).all() and torch.isfinite(lower.last_policy_input).all()
            and torch.isfinite(left_sensor.data.force_matrix_w).all() and torch.isfinite(right_sensor.data.force_matrix_w).all()
        )
        record = {
            "frame": frame_index, "time_s": (frame_index + 1) * dt, "fsm_state": fsm_state.value,
            "subphase": subphase, "failure_unload": failure_unload, "finite": finite,
            "left_contact": left_contact, "right_contact": right_contact,
            "left_force_n": left_force, "right_force_n": right_force,
            "left_impulse_ns": tracker["left_impulse"], "right_impulse_ns": tracker["right_impulse"],
            "combined_impulse_ns": tracker["left_impulse"] + tracker["right_impulse"],
            "contact_force_peak_n": max(left_force, right_force), "contact_force_rate_nps": force_rate,
            "left_contact_onset_frame": tracker["left_onset"], "right_contact_onset_frame": tracker["right_onset"],
            "bilateral_contact_onset_frame": tracker["bilateral_onset"],
            "left_contact_loss_frames": list(tracker["left_losses"]), "right_contact_loss_frames": list(tracker["right_losses"]),
            "robot_box_overlap_count": len(hits), "robot_box_overlap_paths": hits,
            "forbidden_contact_links": forbidden, "approach_displacement_m": approach_displacement,
            "left_actual_surface_gap_m": float(config["object"]["rear_face_xO_m"] - palm_object_pos[left_frame, 0] - config["precontact"]["palm_collision_support_offset_m"]),
            "right_actual_surface_gap_m": float(config["object"]["rear_face_xO_m"] - palm_object_pos[right_frame, 0] - config["precontact"]["palm_collision_support_offset_m"]),
            "left_position_error_m": float(torch.linalg.vector_norm(actual_positions[left_frame] - desired_positions[left_frame]).item()),
            "right_position_error_m": float(torch.linalg.vector_norm(actual_positions[right_frame] - desired_positions[right_frame]).item()),
            "left_orientation_error_deg": float(quaternion_error_deg(actual_quaternions, desired_quaternions)[left_frame]),
            "right_orientation_error_deg": float(quaternion_error_deg(actual_quaternions, desired_quaternions)[right_frame]),
            "minimum_arm_joint_limit_margin_rad": float(torch.minimum(lower_margin, upper_margin).min().item()),
            "arm_torque_ratio_max": float((robot.data.applied_torque[0, arm_ids].abs() / effort_limits).max().item()),
            "base_xy_excursion_m": float(torch.linalg.vector_norm(robot.data.root_link_pos_w[0, :2] - root_reference[:2]).item()),
            "root_height_m": float(robot.data.root_link_pos_w[0, 2]), "root_roll_rad": root_roll,
            "root_pitch_rad": root_pitch, "root_tilt_deg": math.degrees(root_tilt),
            "box_translation_m": float(torch.linalg.vector_norm(box.data.root_link_pos_w[0] - box_attach_reference).item()),
            "box_yaw_change_rad": wrapped_abs(box_yaw, box_yaw_reference),
            "box_linear_speed_mps": float(torch.linalg.vector_norm(box.data.root_com_lin_vel_w[0]).item()),
            "box_angular_speed_radps": float(torch.linalg.vector_norm(box.data.root_com_ang_vel_w[0]).item()),
            "post_initial_reset_count": 0,
        }
        if not finite:
            raise RuntimeError("NONFINITE")
        trace.write(json.dumps(record, sort_keys=True) + "\n")
        frame_index += 1
        RUNTIME_STATE["observed_frames"] = frame_index
        status["observed_frames"] = frame_index
        tracker["last_image"] = Image.fromarray(camera.data.output["rgb"][0].detach().cpu().numpy().astype("uint8"))
        if frame_index % 25 == 0:
            print(f"PHASE={fsm_state.value} step={frame_index}", flush=True)
        return record

    def safety_failure(record: dict, *, precontact_phase: bool, contact_phase: bool) -> str | None:
        if record["forbidden_contact_links"]:
            return "PRECONTACT_COLLISION" if precontact_phase else "FORBIDDEN_BODY_BOX_COLLISION"
        if precontact_phase and (record["left_contact"] or record["right_contact"] or record["robot_box_overlap_count"] > 0):
            return "PRECONTACT_COLLISION"
        gate = config["contact_gate"]
        if (
            record["contact_force_peak_n"] > gate["per_palm_force_peak_threshold_n"]
            or record["left_impulse_ns"] > gate["per_palm_impulse_threshold_ns"]
            or record["right_impulse_ns"] > gate["per_palm_impulse_threshold_ns"]
            or record["combined_impulse_ns"] > gate["excessive_combined_impulse_threshold_ns"]
            or record["contact_force_rate_nps"] > gate["contact_force_rate_threshold_nps"]
        ):
            return "EXCESSIVE_CONTACT_IMPULSE"
        acceptance = config["acceptance"]
        if record["root_height_m"] < acceptance["minimum_root_height_m"]:
            return "ROBOT_FALL"
        if record["root_tilt_deg"] > acceptance["maximum_root_tilt_deg"]:
            return "ROBOT_BAD_TILT"
        if record["minimum_arm_joint_limit_margin_rad"] < acceptance["minimum_arm_joint_limit_margin_rad"]:
            return "JOINT_LIMIT_VIOLATION"
        if record["arm_torque_ratio_max"] > acceptance["maximum_arm_torque_ratio"]:
            return "TORQUE_LIMIT_VIOLATION"
        if contact_phase and (
            record["box_translation_m"] > acceptance["maximum_box_translation_m"]
            or record["box_linear_speed_mps"] > acceptance["maximum_box_linear_speed_mps"]
        ):
            return "OBJECT_TRANSLATION_DURING_ATTACH"
        if contact_phase and (
            record["box_yaw_change_rad"] > acceptance["maximum_box_yaw_change_rad"]
            or record["box_angular_speed_radps"] > acceptance["maximum_box_angular_speed_radps"]
        ):
            return "OBJECT_YAW_DURING_ATTACH"
        return None

    failure_reason = None
    approach_displacement = 0.0
    approach_speed = 0.0
    approach_acceleration = 0.0
    frozen_contact_displacement = None
    bilateral_verify_count = 0
    left_loss_streak = right_loss_streak = 0
    single_hand_count = 0

    with (RUN / "trace.jsonl").open("w", encoding="utf-8", buffering=1) as trace:
        for _ in range(int(config["motion"]["stand_settle_steps"])):
            record = step_record(trace, baseline_positions, baseline_quaternions, "STAND_SETTLE", 0.0)
            failure_reason = safety_failure(record, precontact_phase=False, contact_phase=False)
            if failure_reason:
                transition(AttachState.FAIL, failure_reason); break

        if failure_reason is None:
            transition(AttachState.PRECONTACT)
            move_steps = int(config["motion"]["move_to_precontact_steps"])
            for step in range(move_steps):
                target_positions, target_quaternions = target_pose_in_pelvis(robot, box_fixed_pos, box_fixed_quat, candidate_base, desired_q_object)
                fraction = (step + 1) / move_steps
                desired_positions = baseline_positions + fraction * (target_positions - baseline_positions)
                record = step_record(trace, desired_positions, target_quaternions, "MOVE_TO_PRECONTACT", 0.0)
                failure_reason = safety_failure(record, precontact_phase=True, contact_phase=False)
                if failure_reason:
                    transition(AttachState.FAIL, failure_reason); break
            if failure_reason is None:
                for _ in range(int(config["motion"]["precontact_hold_steps"])):
                    target_positions, target_quaternions = target_pose_in_pelvis(robot, box_fixed_pos, box_fixed_quat, candidate_base, desired_q_object)
                    record = step_record(trace, target_positions, target_quaternions, "PRECONTACT_HOLD", 0.0)
                    failure_reason = safety_failure(record, precontact_phase=True, contact_phase=False)
                    if failure_reason:
                        transition(AttachState.FAIL, failure_reason); break

        if failure_reason is None:
            box_attach_reference = box.data.root_link_pos_w[0].clone()
            _, _, box_yaw_reference, _ = quaternion_rpy_tilt(box.data.root_link_quat_w[0])
            transition(AttachState.APPROACH_NORMAL)
            for _ in range(int(config["motion"]["maximum_approach_steps"])):
                approach_displacement, approach_speed, approach_acceleration = advance_rate_limited(
                    approach_displacement, approach_speed, approach_acceleration,
                    dt_s=dt, speed_limit_mps=config["motion"]["approach_speed_mps"],
                    acceleration_limit_mps2=config["motion"]["approach_acceleration_limit_mps2"],
                    jerk_limit_mps3=config["motion"]["approach_jerk_limit_mps3"],
                )
                candidate = dict(candidate_base)
                candidate["precontact_gap_m"] = candidate_base["precontact_gap_m"] - approach_displacement
                target_positions, target_quaternions = target_pose_in_pelvis(robot, box_fixed_pos, box_fixed_quat, candidate, desired_q_object)
                record = step_record(trace, target_positions, target_quaternions, "NORMAL_RATE_LIMITED", approach_displacement)
                failure_reason = safety_failure(record, precontact_phase=False, contact_phase=record["left_contact"] or record["right_contact"])
                if failure_reason:
                    transition(AttachState.FAIL, failure_reason); break
                if record["left_contact"] or record["right_contact"]:
                    transition(AttachState.BILATERAL_CONTACT_VERIFY); break
                if approach_displacement >= float(config["motion"]["maximum_approach_distance_m"]):
                    failure_reason = "BILATERAL_CONTACT_TIMEOUT"
                    transition(AttachState.FAIL, failure_reason); break

        if failure_reason is None and fsm_state is AttachState.BILATERAL_CONTACT_VERIFY:
            remaining = int(config["motion"]["maximum_approach_steps"])
            for _ in range(remaining):
                if frozen_contact_displacement is None:
                    approach_displacement, approach_speed, approach_acceleration = advance_rate_limited(
                        approach_displacement, approach_speed, approach_acceleration,
                        dt_s=dt, speed_limit_mps=config["motion"]["approach_speed_mps"],
                        acceleration_limit_mps2=config["motion"]["approach_acceleration_limit_mps2"],
                        jerk_limit_mps3=config["motion"]["approach_jerk_limit_mps3"],
                    )
                commanded_displacement = approach_displacement if frozen_contact_displacement is None else frozen_contact_displacement
                candidate = dict(candidate_base)
                candidate["precontact_gap_m"] = candidate_base["precontact_gap_m"] - commanded_displacement
                target_positions, target_quaternions = target_pose_in_pelvis(robot, box_fixed_pos, box_fixed_quat, candidate, desired_q_object)
                record = step_record(trace, target_positions, target_quaternions, "BILATERAL_VERIFY", commanded_displacement)
                failure_reason = safety_failure(record, precontact_phase=False, contact_phase=True)
                if failure_reason:
                    transition(AttachState.FAIL, failure_reason); break
                if record["left_contact"] and record["right_contact"]:
                    if frozen_contact_displacement is None:
                        frozen_contact_displacement = commanded_displacement
                    bilateral_verify_count += 1
                    single_hand_count = 0
                else:
                    bilateral_verify_count = 0
                    single_hand_count += 1
                    if single_hand_count > int(config["motion"]["single_hand_maximum_steps"]):
                        failure_reason = (
                            "RIGHT_CONTACT_MISSING" if tracker["left_onset"] is not None and tracker["right_onset"] is None
                            else "LEFT_CONTACT_MISSING" if tracker["right_onset"] is not None and tracker["left_onset"] is None
                            else "BILATERAL_CONTACT_TIMEOUT"
                        )
                        transition(AttachState.FAIL, failure_reason); break
                if bilateral_verify_count >= int(config["motion"]["bilateral_verification_steps"]):
                    transition(AttachState.ATTACHED_HOLD); break
                if commanded_displacement >= float(config["motion"]["maximum_approach_distance_m"]):
                    failure_reason = "BILATERAL_CONTACT_TIMEOUT"
                    transition(AttachState.FAIL, failure_reason); break

        if failure_reason is None and fsm_state is AttachState.ATTACHED_HOLD:
            assert frozen_contact_displacement is not None
            candidate = dict(candidate_base)
            candidate["precontact_gap_m"] = candidate_base["precontact_gap_m"] - frozen_contact_displacement
            for _ in range(int(config["motion"]["attached_hold_steps"])):
                target_positions, target_quaternions = target_pose_in_pelvis(robot, box_fixed_pos, box_fixed_quat, candidate, desired_q_object)
                record = step_record(trace, target_positions, target_quaternions, "NO_FURTHER_PENETRATION_HOLD", frozen_contact_displacement)
                failure_reason = safety_failure(record, precontact_phase=False, contact_phase=True)
                if failure_reason:
                    transition(AttachState.FAIL, failure_reason); break
                left_loss_streak = 0 if record["left_contact"] else left_loss_streak + 1
                right_loss_streak = 0 if record["right_contact"] else right_loss_streak + 1
                if left_loss_streak > int(config["motion"]["contact_loss_grace_steps"]):
                    failure_reason = "LEFT_CONTACT_LOST"; transition(AttachState.FAIL, failure_reason); break
                if right_loss_streak > int(config["motion"]["contact_loss_grace_steps"]):
                    failure_reason = "RIGHT_CONTACT_LOST"; transition(AttachState.FAIL, failure_reason); break
            if failure_reason is None:
                transition(AttachState.PASS)

        if failure_reason is not None:
            pre_positions, pre_quaternions = target_pose_in_pelvis(robot, box_fixed_pos, box_fixed_quat, candidate_base, desired_q_object)
            for _ in range(int(config["motion"]["failure_unload_steps"])):
                step_record(trace, pre_positions, pre_quaternions, "FAILURE_UNLOAD_TO_PRECONTACT", 0.0, failure_unload=True)

    print("PHASE=FINAL_AUDIT", flush=True)
    if tracker["last_image"] is None:
        raise RuntimeError("MISSING_FINAL_IMAGE")
    tracker["last_image"].save(RUN / "final.png")
    if failure_reason is not None:
        tracker["last_image"].save(RUN / "failure.png")
    outcome = {
        "schema_version": 1, "stage": "S2-03", "terminal_state": fsm_state.value,
        "failure_reason": failure_reason, "fsm_transitions": transitions,
        "left_contact_onset_frame": tracker["left_onset"], "right_contact_onset_frame": tracker["right_onset"],
        "bilateral_contact_onset_frame": tracker["bilateral_onset"],
        "left_contact_loss_frames": tracker["left_losses"], "right_contact_loss_frames": tracker["right_losses"],
        "left_impulse_ns": tracker["left_impulse"], "right_impulse_ns": tracker["right_impulse"],
        "observed_frames": frame_index, "box_push_commanded": False,
    }
    write_json(RUN / "fsm_outcome.json", outcome)
    status.update({"status": "COMPLETE", "primary_reason": None, "observed_frames": frame_index})
    write_json(RUN / "runner_status.json", status)


if __name__ == "__main__":
    exit_code = 0
    main_completed = False
    try:
        main()
        main_completed = True
    except BaseException as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        write_implementation_exception(RUN / "implementation_exception.json", exc, {**RUNTIME_STATE, "exception_phase": "MAIN"})
        write_json(RUN / "runner_status.json", {
            "status": "INVALID", "primary_reason": "IMPLEMENTATION_EXCEPTION",
            "environment_created": RUNTIME_STATE["environment_created"],
            "observed_frames": RUNTIME_STATE["observed_frames"],
            "multiple_isaac_processes": False, "error": repr(exc),
        })
        exit_code = 1
    try:
        if RUNTIME_ENV is not None:
            RUNTIME_ENV.close()
        simulation_app.close()
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        if main_completed:
            write_json(RUN / "teardown_status.json", {
                "schema_version": 1, "status": "FAIL",
                "primary_reason": "KIT_TEARDOWN_EXCEPTION_AFTER_AUTHORITATIVE_RESULT",
                "error": repr(exc),
            })
        elif exit_code == 0:
            write_implementation_exception(RUN / "implementation_exception.json", exc, {**RUNTIME_STATE, "exception_phase": "TEARDOWN"})
            exit_code = 1
    raise SystemExit(exit_code)
