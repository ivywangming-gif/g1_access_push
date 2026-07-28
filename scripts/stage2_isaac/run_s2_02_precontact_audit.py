#!/usr/bin/env python3
"""Run S2-02 development search/preflight or one formal no-contact episode."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

from g1_access_push.stage2.s2_01_process import (
    validate_controller_checkpoint,
    validate_resolved_sensor_body,
    validate_robot_filter_tensor,
    write_implementation_exception,
)

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
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

from agile.rl_env.assets.robots.unitree_g1 import G1_W_HANDS_AGILE_ACTION_SCALE  # noqa: E402
from g1_access_push.sim.stage1.no_box_env_cfg import (  # noqa: E402
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)
from g1_access_push.sim.stage2.s2_02_precontact_env import (  # noqa: E402
    BOX_SENSOR_PRIM_PATH,
    ROBOT_FILTER_EXPRESSIONS,
    build_s2_02_env_cfg,
)
from g1_access_push.stage2.s2_02_contract import (  # noqa: E402
    candidate_static_checks,
    load_config,
    object_local_targets,
    score_candidate,
    select_candidate,
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
        if "run_s2_02_precontact_audit.py" in command:
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
    inertias[..., 0] = float(diagonal[0])
    inertias[..., 4] = float(diagonal[1])
    inertias[..., 8] = float(diagonal[2])
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
        "runtime_view": type(view).__qualname__,
        "getter_shapes": {"masses": list(actual_mass.shape), "coms": list(actual_com.shape), "inertias": list(actual_inertia.shape)},
        "mass_kg": float(actual_mass.reshape(-1)[0]),
        "com_local_xyz_m": [float(value) for value in first_com[:3]],
        "principal_axes_xyzw": [float(value) for value in first_com[3:7]],
        "diagonal_inertia_kg_m2": [float(first_inertia[index]) for index in (0, 4, 8)],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def overlap_query(box_pos: torch.Tensor, box_quat: torch.Tensor, half_extents: list[float]) -> dict:
    hits: list[dict[str, str]] = []

    def callback(hit) -> bool:
        collision = str(getattr(hit, "collision", ""))
        rigid_body = str(getattr(hit, "rigid_body", ""))
        if collision.startswith(ROBOT_PRIM_PATH) or rigid_body.startswith(ROBOT_PRIM_PATH):
            hits.append({"collision": collision, "rigid_body": rigid_body})
        return True

    get_physx_scene_query_interface().overlap_box(
        carb.Float3(*half_extents),
        carb.Float3(*[float(value) for value in box_pos]),
        carb.Float4(float(box_quat[1]), float(box_quat[2]), float(box_quat[3]), float(box_quat[0])),
        callback,
        False,
    )
    return {"robot_hit_count": len(hits), "robot_hits": hits}


def place_box(box, center_x: float, center_y: float, center_z: float) -> None:
    state = box.data.default_root_state.clone()
    state[0, :3] = torch.tensor([center_x, center_y, center_z], device=state.device)
    state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device)
    state[0, 7:] = 0.0
    box.write_root_state_to_sim(state)


def target_pose_in_pelvis(robot, box_pos_one: torch.Tensor, box_quat_one: torch.Tensor, candidate: dict, desired_q_object: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local_positions = torch.tensor(object_local_targets(candidate), device=robot.device, dtype=torch.float32)
    target_world_pos, target_world_quat = math_utils.combine_frame_transforms(
        box_pos_one.expand(2, 3), box_quat_one.expand(2, 4),
        local_positions, desired_q_object.expand(2, 4),
    )
    pelvis_pos = robot.data.root_link_pos_w[0].expand(2, 3)
    pelvis_quat = robot.data.root_link_quat_w[0].expand(2, 4)
    return math_utils.subtract_frame_transforms(pelvis_pos, pelvis_quat, target_world_pos, target_world_quat)


def main() -> None:
    global RUNTIME_ENV
    RUN.mkdir(parents=True, exist_ok=True)
    run_mode = "preflight" if args.preflight_only else "formal"
    expected_frames = int(args.preflight_steps) if args.preflight_only else None
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
    if resolved.get("config_sha256") != sha256_file(args.config):
        raise RuntimeError("CONFIG_SHA_MISMATCH")
    s2_01 = Path(config["certification"]["s2_01_result_path"])
    if not s2_01.is_file() or json.loads(s2_01.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("S2_01_PASS_EVIDENCE_MISSING")
    if args.formal and not resolved.get("runnable"):
        raise RuntimeError("FORMAL_REQUIRES_FROZEN_SELECTION")

    cfg = build_s2_02_env_cfg()
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
    contact = env.scene["box_robot_contact"]
    camera = env.scene["audit_camera"]
    lower = env.action_manager.get_term("lower_body_joint_pos")

    checkpoint = Path(lower.cfg.policy_path).resolve()
    checkpoint_audit = validate_controller_checkpoint(
        checkpoint,
        sha256_file(checkpoint),
        config["certification"]["controller_checkpoint_sha256"],
        [config["certification"]["forbidden_checkpoint_sha256"]],
        ["model_1999.pt"],
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

    box_prim = stage.GetPrimAtPath(BOX_PRIM_PATH)
    force_matrix = contact.data.force_matrix_w
    body_audit = validate_resolved_sensor_body(
        BOX_SENSOR_PRIM_PATH, contact.cfg.prim_path, list(contact.body_names), contact.num_bodies,
        initialized=contact.is_initialized,
        contact_reporter_enabled=bool(box_prim.HasAPI(PhysxSchema.PhysxContactReportAPI)),
        rigid_body_bound=bool(box_prim.HasAPI(UsdPhysics.RigidBodyAPI)),
        data_available=force_matrix is not None,
    )
    robot_rigid_bodies = [
        str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    filter_audit = validate_robot_filter_tensor(
        list(ROBOT_FILTER_EXPRESSIONS), int(contact.contact_physx_view.filter_count),
        None if force_matrix is None else list(force_matrix.shape), num_envs=1,
        box_body_count=contact.num_bodies, usd_candidate_robot_rigid_body_count=len(robot_rigid_bodies),
        force_matrix_available=force_matrix is not None,
        force_matrix_finite=bool(force_matrix is not None and torch.isfinite(force_matrix).all()),
    )
    contact_audit = {**body_audit, **filter_audit}
    contact_audit["status"] = "PASS" if body_audit["sensor_body_audit_pass"] and filter_audit["filter_tensor_initialization_pass"] else "FAIL"
    write_json(RUN / "contact_sensor_audit.json", contact_audit)
    if contact_audit["status"] != "PASS":
        raise RuntimeError("CONTACT_SENSOR_INITIALIZATION_FAILED")

    frame_names = list(palms.data.target_frame_names)
    left_frame = frame_names.index("left_hand_palm")
    right_frame = frame_names.index("right_hand_palm")
    if (left_frame, right_frame) != (0, 1):
        raise RuntimeError(f"PALM_FRAME_ORDER_MISMATCH:{frame_names}")
    arm_ids, arm_names = robot.find_joints(LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES)
    palm_ids, palm_body_names = robot.find_bodies(["left_hand_palm_link", "right_hand_palm_link"], preserve_order=True)
    joint_limits = robot.data.joint_pos_limits[0, arm_ids].clone()
    effort_limits = robot.data.joint_effort_limits[0, arm_ids].abs().clamp_min(1.0e-6)
    robot_collision_paths = [
        str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH))
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    runtime_geometry = {
        "pelvis_pose_world": [float(value) for value in torch.cat((robot.data.root_link_pos_w[0], robot.data.root_link_quat_w[0]))],
        "palm_frame_names": frame_names,
        "palm_body_names": palm_body_names,
        "palm_body_ids": palm_ids,
        "palm_positions_in_pelvis_m": palms.data.target_pos_source[0].detach().cpu().tolist(),
        "palm_quaternions_in_pelvis_wxyz": palms.data.target_quat_source[0].detach().cpu().tolist(),
        "arm_joint_names": arm_names,
        "arm_joint_positions_rad": robot.data.joint_pos[0, arm_ids].detach().cpu().tolist(),
        "arm_joint_limits_rad": joint_limits.detach().cpu().tolist(),
        "robot_rigid_body_paths": robot_rigid_bodies,
        "robot_collision_paths": robot_collision_paths,
        "robot_body_names": list(robot.body_names),
        "robot_body_link_positions_world_m": robot.data.body_link_pos_w[0].detach().cpu().tolist(),
        "box_pose_world": [float(value) for value in torch.cat((box.data.root_link_pos_w[0], box.data.root_link_quat_w[0]))],
        "box_rear_face_local_x_m": float(config["object"]["rear_face_xO_m"]),
        "palm_normal_axis_source": "G1_PALM_LINK_LOCAL_PLUS_Z_VALIDATED_AGAINST_S1_07_BASELINE_AND_RUNTIME",
        "palm_local_normal_axis": config["controller"]["palm_local_normal_axis"],
    }
    write_json(RUN / "runtime_geometry_audit.json", runtime_geometry)
    write_json(RUN / "controller_contract_audit.json", {
        **checkpoint_audit,
        "recurrent_reset": recurrent_reset,
        "action_contract": action_contract,
        "lower_body_scale_source": "G1_W_HANDS_AGILE_ACTION_SCALE",
        "lower_body_scale_source_entry_count": len(G1_W_HANDS_AGILE_ACTION_SCALE),
        "resolved_lower_body_scale": [float(value) for value in lower._policy_output_scale[0]],
        "resolved_lower_body_offset": [float(value) for value in lower._policy_output_offset[0]],
        "lower_body_command": config["controller"]["lower_body_command"],
    })

    desired_q_object = torch.tensor(config["controller"]["desired_palm_quaternion_in_object_wxyz"], device=env.device, dtype=torch.float32)
    local_normal = torch.tensor(config["controller"]["palm_local_normal_axis"], device=env.device, dtype=torch.float32)
    maximum_position = float(config["controller"]["maximum_position_correction_m"])
    maximum_orientation = float(config["controller"]["maximum_orientation_correction_rad"])
    baseline_root_pos = robot.data.root_link_pos_w[0].clone()
    development_search = args.preflight_only and config["selection"]["status"] != "FROZEN"
    if development_search:
        place_box(
            box, float(baseline_root_pos[0]) + 3.0, float(baseline_root_pos[1]),
            float(config["object"]["spawn_center_z_m"]),
        )

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

    for warmup_step in range(int(config["search"]["warmup_steps"])):
        actions.zero_()
        actions[:, lower_slice] = lower_command
        env.step(actions)
        if (warmup_step + 1) % 25 == 0:
            print(f"PHASE=warmup step={warmup_step + 1}", flush=True)

    baseline_positions = palms.data.target_pos_source[0].clone()
    baseline_quaternions = palms.data.target_quat_source[0].clone()
    baseline_arm_positions = robot.data.joint_pos[0, arm_ids].clone()
    camera.set_world_poses_from_view(
        torch.tensor([[2.0, -2.0, 1.55]], device=env.device),
        torch.tensor([[0.45, 0.0, 0.70]], device=env.device),
    )

    def sample(candidate: dict, desired_positions: torch.Tensor, desired_quaternions: torch.Tensor, root_reference: torch.Tensor, box_reference: torch.Tensor, yaw_reference: float, geometry_box_pos: torch.Tensor, geometry_box_quat: torch.Tensor) -> dict:
        actual_positions = palms.data.target_pos_source[0]
        actual_quaternions = palms.data.target_quat_source[0]
        palm_world_pos, palm_world_quat = math_utils.combine_frame_transforms(
            robot.data.root_link_pos_w[0].expand(2, 3), robot.data.root_link_quat_w[0].expand(2, 4),
            actual_positions, actual_quaternions,
        )
        palm_object_pos, _ = math_utils.subtract_frame_transforms(
            geometry_box_pos.expand(2, 3), geometry_box_quat.expand(2, 4),
            palm_world_pos, palm_world_quat,
        )
        object_pos_pelvis, object_quat_pelvis = math_utils.subtract_frame_transforms(
            robot.data.root_link_pos_w[0].unsqueeze(0), robot.data.root_link_quat_w[0].unsqueeze(0),
            geometry_box_pos.unsqueeze(0), geometry_box_quat.unsqueeze(0),
        )
        del object_pos_pelvis
        object_x_pelvis = math_utils.quat_apply(object_quat_pelvis.expand(2, 4), torch.tensor([[1.0, 0.0, 0.0]], device=env.device).expand(2, 3))
        palm_normals = math_utils.quat_apply(actual_quaternions, local_normal.expand(2, 3))
        normal_dots = torch.sum(palm_normals * object_x_pelvis, dim=-1)
        forces = torch.linalg.vector_norm(contact.data.force_matrix_w[0, 0], dim=-1)
        overlap = overlap_query(geometry_box_pos, geometry_box_quat, [0.6, 0.3, 0.6])
        arm_pos = robot.data.joint_pos[0, arm_ids]
        arm_target = robot.data.joint_pos_target[0, arm_ids]
        lower_margin = arm_pos - joint_limits[:, 0]
        upper_margin = joint_limits[:, 1] - arm_pos
        root_roll, root_pitch, _, root_tilt = quaternion_rpy_tilt(robot.data.root_link_quat_w[0])
        _, _, box_yaw, _ = quaternion_rpy_tilt(box.data.root_link_quat_w[0])
        box_translation = float(torch.linalg.vector_norm(box.data.root_link_pos_w[0] - box_reference).item())
        rear_local = torch.tensor([[-0.6, 0.0, 0.0]], device=env.device)
        rear_world, _ = math_utils.combine_frame_transforms(
            geometry_box_pos.unsqueeze(0), geometry_box_quat.unsqueeze(0),
            rear_local, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device),
        )
        values = [
            *[float(value) for value in actual_positions.flatten()],
            *[float(value) for value in actual_quaternions.flatten()],
            *[float(value) for value in arm_pos],
            *[float(value) for value in box.data.root_link_pos_w[0]],
            *[float(value) for value in robot.data.root_link_pos_w[0]],
        ]
        return {
            "finite": bool(all(math.isfinite(value) for value in values) and torch.isfinite(lower.last_policy_input).all() and torch.isfinite(lower._policy_actions).all()),
            "robot_box_contact_force_n": float(forces.max().item()) if forces.numel() else 0.0,
            "robot_box_overlap_count": int(overlap["robot_hit_count"]),
            "robot_box_overlap_paths": overlap["robot_hits"],
            "left_position_error_m": float(torch.linalg.vector_norm(actual_positions[left_frame] - desired_positions[left_frame]).item()),
            "right_position_error_m": float(torch.linalg.vector_norm(actual_positions[right_frame] - desired_positions[right_frame]).item()),
            "left_orientation_error_deg": float(quaternion_error_deg(actual_quaternions, desired_quaternions)[left_frame].item()),
            "right_orientation_error_deg": float(quaternion_error_deg(actual_quaternions, desired_quaternions)[right_frame].item()),
            "left_palm_normal_alignment_dot": float(normal_dots[left_frame].item()),
            "right_palm_normal_alignment_dot": float(normal_dots[right_frame].item()),
            "left_actual_gap_m": float(config["object"]["rear_face_xO_m"] - palm_object_pos[left_frame, 0]),
            "right_actual_gap_m": float(config["object"]["rear_face_xO_m"] - palm_object_pos[right_frame, 0]),
            "commanded_precontact_gap_m": float(candidate["precontact_gap_m"]),
            "box_rear_face_position_world_m": [float(value) for value in rear_world[0]],
            "minimum_arm_joint_limit_margin_rad": float(torch.minimum(lower_margin, upper_margin).min().item()),
            "arm_joint_target_error_max_rad": float((arm_pos - arm_target).abs().max().item()),
            "arm_torque_ratio_max": float((robot.data.applied_torque[0, arm_ids].abs() / effort_limits).max().item()),
            "base_xy_excursion_m": float(torch.linalg.vector_norm(robot.data.root_link_pos_w[0, :2] - root_reference[:2]).item()),
            "root_height_m": float(robot.data.root_link_pos_w[0, 2]),
            "root_roll_rad": root_roll,
            "root_pitch_rad": root_pitch,
            "root_tilt_deg": math.degrees(root_tilt),
            "box_translation_m": box_translation,
            "box_yaw_change_rad": wrapped_abs(box_yaw, yaw_reference),
            "box_linear_speed_mps": float(torch.linalg.vector_norm(box.data.root_com_lin_vel_w[0]).item()),
            "box_angular_speed_radps": float(torch.linalg.vector_norm(box.data.root_com_ang_vel_w[0]).item()),
            "post_initial_reset_count": 0,
        }

    candidate_records: list[dict] = []
    if args.preflight_only:
        selection = config["selection"]
        if selection["status"] == "FROZEN":
            candidates = [{key: selection[key] for key in ("contact_height_m", "tangential_separation_m", "base_to_box_center_distance_m", "precontact_gap_m")}]
            candidate_indices = [int(selection["candidate_index"])]
        else:
            candidates = list(config["search"]["candidates"])
            candidate_indices = list(range(1, len(candidates) + 1))
        for candidate_index, candidate in zip(candidate_indices, candidates, strict=True):
            static_checks = candidate_static_checks(candidate, config["object"])
            center_x = float(baseline_root_pos[0]) + float(candidate["base_to_box_center_distance_m"])
            center_y = float(baseline_root_pos[1])
            geometry_box_pos = torch.tensor(
                [center_x, center_y, float(config["object"]["spawn_center_z_m"])], device=env.device
            )
            geometry_box_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            if not development_search:
                place_box(box, center_x, center_y, float(config["object"]["spawn_center_z_m"]))
            for _ in range(2):
                command(baseline_positions, baseline_quaternions)
            root_reference = robot.data.root_link_pos_w[0].clone()
            box_reference = box.data.root_link_pos_w[0].clone()
            _, _, yaw_reference, _ = quaternion_rpy_tilt(box.data.root_link_quat_w[0])
            samples: list[dict] = []
            transition_steps = int(config["search"]["transition_steps"])
            for step in range(transition_steps):
                target_positions, target_quaternions = target_pose_in_pelvis(robot, geometry_box_pos, geometry_box_quat, candidate, desired_q_object)
                fraction = (step + 1) / transition_steps
                desired_positions = baseline_positions + fraction * (target_positions - baseline_positions)
                command(desired_positions, target_quaternions)
                samples.append(sample(candidate, desired_positions, target_quaternions, root_reference, box_reference, yaw_reference, geometry_box_pos, geometry_box_quat))
                if (step + 1) % 25 == 0 or step + 1 == transition_steps:
                    print(f"PHASE=candidate index={candidate_index} transition_step={step + 1}", flush=True)
            hold_samples: list[dict] = []
            for step in range(int(config["search"]["hold_steps"])):
                desired_positions, desired_quaternions = target_pose_in_pelvis(robot, geometry_box_pos, geometry_box_quat, candidate, desired_q_object)
                command(desired_positions, desired_quaternions)
                current = sample(candidate, desired_positions, desired_quaternions, root_reference, box_reference, yaw_reference, geometry_box_pos, geometry_box_quat)
                samples.append(current)
                hold_samples.append(current)
            position_errors = [item[side] for item in hold_samples for side in ("left_position_error_m", "right_position_error_m")]
            left_errors = [item["left_position_error_m"] for item in hold_samples]
            right_errors = [item["right_position_error_m"] for item in hold_samples]
            metrics = {
                "maximum_robot_box_contact_force_n": max(item["robot_box_contact_force_n"] for item in samples),
                "maximum_robot_box_overlap_count": max(item["robot_box_overlap_count"] for item in samples),
                "maximum_hand_position_error_m": max(position_errors),
                "left_right_position_error_asymmetry_m": abs(sum(left_errors) / len(left_errors) - sum(right_errors) / len(right_errors)),
                "minimum_palm_normal_alignment_dot": min(min(item["left_palm_normal_alignment_dot"], item["right_palm_normal_alignment_dot"]) for item in hold_samples),
                "minimum_actual_gap_m": min(min(item["left_actual_gap_m"], item["right_actual_gap_m"]) for item in hold_samples),
                "maximum_actual_gap_error_m": max(max(abs(item["left_actual_gap_m"] - candidate["precontact_gap_m"]), abs(item["right_actual_gap_m"] - candidate["precontact_gap_m"])) for item in hold_samples),
                "minimum_arm_joint_limit_margin_rad": min(item["minimum_arm_joint_limit_margin_rad"] for item in samples),
                "arm_joint_target_p95_error_rad": float(torch.quantile(torch.tensor([item["arm_joint_target_error_max_rad"] for item in samples]), 0.95).item()),
                "maximum_arm_torque_ratio": max(item["arm_torque_ratio_max"] for item in samples),
                "maximum_base_xy_excursion_m": max(item["base_xy_excursion_m"] for item in samples),
                "minimum_root_height_m": min(item["root_height_m"] for item in samples),
                "maximum_root_height_m": max(item["root_height_m"] for item in samples),
                "maximum_root_tilt_deg": max(item["root_tilt_deg"] for item in samples),
                "maximum_box_translation_m": max(item["box_translation_m"] for item in hold_samples),
                "maximum_box_yaw_change_rad": max(item["box_yaw_change_rad"] for item in hold_samples),
                "maximum_box_linear_speed_mps": max(item["box_linear_speed_mps"] for item in hold_samples),
                "maximum_box_angular_speed_radps": max(item["box_angular_speed_radps"] for item in hold_samples),
                "arm_displacement_norm_rad": float(torch.linalg.vector_norm(robot.data.joint_pos[0, arm_ids] - baseline_arm_positions).item()),
            }
            acceptance = config["acceptance"]
            checks = {
                **static_checks,
                "finite": all(item["finite"] for item in samples),
                "no_contact": metrics["maximum_robot_box_contact_force_n"] <= acceptance["maximum_robot_box_contact_force_n"],
                "no_overlap": metrics["maximum_robot_box_overlap_count"] <= acceptance["maximum_robot_box_overlap_count"],
                "reachable": metrics["maximum_hand_position_error_m"] <= acceptance["maximum_hand_position_max_m"],
                "normal_alignment": metrics["minimum_palm_normal_alignment_dot"] >= acceptance["minimum_palm_normal_alignment_dot"],
                "joint_margin": metrics["minimum_arm_joint_limit_margin_rad"] >= acceptance["minimum_arm_joint_limit_margin_rad"],
                "symmetric_tracking": metrics["left_right_position_error_asymmetry_m"] <= acceptance["maximum_left_right_position_error_asymmetry_m"],
                "gap_clear": metrics["minimum_actual_gap_m"] >= acceptance["minimum_actual_precontact_gap_m"] and metrics["maximum_actual_gap_error_m"] <= acceptance["maximum_actual_gap_error_m"],
                "arm_tracking": metrics["arm_joint_target_p95_error_rad"] <= acceptance["maximum_arm_joint_target_p95_error_rad"],
                "arm_torque": metrics["maximum_arm_torque_ratio"] <= acceptance["maximum_arm_torque_ratio"],
                "robot_stable": metrics["maximum_base_xy_excursion_m"] <= acceptance["maximum_base_xy_excursion_m"] and metrics["minimum_root_height_m"] >= acceptance["minimum_root_height_m"] and metrics["maximum_root_height_m"] <= acceptance["maximum_root_height_m"] and metrics["maximum_root_tilt_deg"] <= acceptance["maximum_root_tilt_deg"],
                "box_still": metrics["maximum_box_translation_m"] <= acceptance["maximum_box_translation_m"] and metrics["maximum_box_yaw_change_rad"] <= acceptance["maximum_box_yaw_change_rad"] and metrics["maximum_box_linear_speed_mps"] <= acceptance["maximum_box_linear_speed_mps"] and metrics["maximum_box_angular_speed_radps"] <= acceptance["maximum_box_angular_speed_radps"],
            }
            record = {
                "candidate_index": candidate_index,
                "candidate": candidate,
                "object_local_targets_m": object_local_targets(candidate),
                "desired_palm_quaternion_in_object_wxyz": config["controller"]["desired_palm_quaternion_in_object_wxyz"],
                "geometry_box_pose_source": "PHYSX_SCENE_QUERY_DEVELOPMENT_POSE" if development_search else "PHYSICAL_BOX_RUNTIME_POSE",
                "checks": checks,
                "metrics": metrics,
                "failed_checks": [name for name, passed in checks.items() if not passed],
                "hold_final_sample": hold_samples[-1],
            }
            record["score"] = list(score_candidate(record))
            record["passed"] = score_candidate(record)[0] == 1.0
            candidate_records.append(record)
            for step in range(int(config["search"]["return_steps"])):
                command(baseline_positions, baseline_quaternions)
                if step + 1 == int(config["search"]["return_steps"]):
                    print(f"PHASE=return index={candidate_index} step={step + 1}", flush=True)
        selected = select_candidate(candidate_records)
        write_json(RUN / "candidate_records.json", candidate_records)
        write_json(RUN / "search_result.json", {
            "schema_version": 1,
            "stage": "S2-02",
            "status": "PASS" if selected is not None else "FAIL",
            "primary_reason": "FEASIBLE_PRECONTACT_GEOMETRY_FOUND" if selected is not None else "NO_FEASIBLE_PRECONTACT_GEOMETRY",
            "candidate_count": len(candidate_records),
            "selected_candidate": selected,
        })
        for _ in range(60):
            command(baseline_positions, baseline_quaternions)

    last_image = None
    if args.preflight_only:
        expected_frames = int(args.preflight_steps)
        with (RUN / "trace.jsonl").open("w", encoding="utf-8", buffering=1) as trace:
            for frame in range(expected_frames):
                actions.zero_()
                actions[:, lower_slice] = lower_command
                env.step(actions)
                record = {"frame": frame, "finite": bool(torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(lower.last_policy_input).all())}
                trace.write(json.dumps(record, sort_keys=True) + "\n")
                RUNTIME_STATE["observed_frames"] = frame + 1
                status["observed_frames"] = frame + 1
                last_image = Image.fromarray(camera.data.output["rgb"][0].detach().cpu().numpy().astype("uint8"))
        image_path = RUN / "preflight.png"
        if last_image is not None:
            last_image.save(image_path)
        search_result = json.loads((RUN / "search_result.json").read_text(encoding="utf-8"))
        frozen = config["selection"]["status"] == "FROZEN"
        checks = {
            "environment_created": True,
            "runtime_mass_properties": mass_audit["status"] == "PASS",
            "contact_sensor": contact_audit["status"] == "PASS",
            "controller_contract": checkpoint_audit["status"] == "PASS" and all(recurrent_reset.values()),
            "runtime_geometry_complete": bool(len(robot.body_names) > 0 and len(arm_names) == 14 and len(palm_body_names) == 2),
            "candidate_sweep_complete": len(candidate_records) == (1 if frozen else len(config["search"]["candidates"])),
            "frozen_candidate_valid": (not frozen) or search_result["status"] == "PASS",
            "trace_writer": sum(1 for line in (RUN / "trace.jsonl").read_text().splitlines() if line.strip()) == expected_frames,
            "camera_output": image_path.is_file() and image_path.stat().st_size > 0,
            "no_second_isaac": not status["multiple_isaac_processes"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        write_json(RUN / "preflight_result.json", {
            "schema_version": 1,
            "stage": "S2-02",
            "mode": "PREFLIGHT_ONLY",
            "status": "PASS" if not failed else "FAIL",
            "primary_reason": "ALL_PREFLIGHT_GATES_PASSED" if not failed else "PREFLIGHT_GATES_FAILED",
            "checks": checks,
            "failed_checks": failed,
            "steps": expected_frames,
            "config_sha256": sha256_file(args.config),
            "resolved_config_sha256": sha256_file(args.resolved_config),
            "scientific_result_created": False,
        })
        status.update({"status": "COMPLETE" if not failed else "PREFLIGHT_FAILED", "primary_reason": None if not failed else "PREFLIGHT_GATES_FAILED", "observed_frames": expected_frames})
        write_json(RUN / "runner_status.json", status)
        return

    selection = config["selection"]
    candidate = {key: selection[key] for key in ("contact_height_m", "tangential_separation_m", "base_to_box_center_distance_m", "precontact_gap_m")}
    place_box(
        box,
        float(baseline_root_pos[0]) + float(candidate["base_to_box_center_distance_m"]),
        float(baseline_root_pos[1]),
        float(config["object"]["spawn_center_z_m"]),
    )
    geometry_box_pos = box.data.root_link_pos_w[0]
    geometry_box_quat = box.data.root_link_quat_w[0]
    box_reference = box.data.root_link_pos_w[0].clone()
    root_reference = robot.data.root_link_pos_w[0].clone()
    _, _, yaw_reference, _ = quaternion_rpy_tilt(box.data.root_link_quat_w[0])
    settle_steps = int(config["formal"]["stand_settle_steps"])
    move_steps = int(config["formal"]["move_to_precontact_steps"])
    hold_steps = int(config["formal"]["precontact_hold_steps"])
    expected_frames = settle_steps + move_steps + hold_steps
    with (RUN / "trace.jsonl").open("w", encoding="utf-8", buffering=1) as trace:
        for frame in range(expected_frames):
            target_positions, target_quaternions = target_pose_in_pelvis(robot, box.data.root_link_pos_w[0], box.data.root_link_quat_w[0], candidate, desired_q_object)
            if frame < settle_steps:
                phase = "STAND_SETTLE"
                desired_positions = baseline_positions
                desired_quaternions = baseline_quaternions
            elif frame < settle_steps + move_steps:
                phase = "MOVE_TO_PRECONTACT"
                fraction = (frame - settle_steps + 1) / move_steps
                desired_positions = baseline_positions + fraction * (target_positions - baseline_positions)
                desired_quaternions = target_quaternions
            else:
                phase = "PRECONTACT_HOLD"
                desired_positions = target_positions
                desired_quaternions = target_quaternions
            command(desired_positions, desired_quaternions)
            record = sample(candidate, desired_positions, desired_quaternions, root_reference, box_reference, yaw_reference, box.data.root_link_pos_w[0], box.data.root_link_quat_w[0])
            record.update({"frame": frame, "time_s": (frame + 1) * env.step_dt, "phase": phase})
            trace.write(json.dumps(record, sort_keys=True) + "\n")
            RUNTIME_STATE["observed_frames"] = frame + 1
            status["observed_frames"] = frame + 1
            if (frame + 1) % 25 == 0 or frame + 1 in (settle_steps, settle_steps + move_steps, expected_frames):
                print(f"PHASE={phase} step={frame + 1}/{expected_frames}", flush=True)
            last_image = Image.fromarray(camera.data.output["rgb"][0].detach().cpu().numpy().astype("uint8"))
    print("PHASE=FINAL_AUDIT", flush=True)
    if last_image is None:
        raise RuntimeError("MISSING_FINAL_IMAGE")
    last_image.save(RUN / "final.png")
    status.update({"status": "COMPLETE", "primary_reason": None, "observed_frames": expected_frames})
    write_json(RUN / "runner_status.json", status)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
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
        if exit_code == 0:
            write_implementation_exception(RUN / "implementation_exception.json", exc, {**RUNTIME_STATE, "exception_phase": "TEARDOWN"})
            exit_code = 1
    raise SystemExit(exit_code)
