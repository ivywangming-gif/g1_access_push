#!/usr/bin/env python3
"""Run exactly one headless S2-01 no-contact standing episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

from g1_access_push.stage2.s2_01_process import (
    validate_controller_checkpoint,
    validate_net_force_tensor,
    validate_resolved_sensor_body,
    validate_robot_filter_tensor,
    write_implementation_exception,
)

bootstrap_parser = argparse.ArgumentParser(add_help=False)
bootstrap_parser.add_argument("--run-root", type=Path, required=True)
bootstrap_args, _ = bootstrap_parser.parse_known_args()
RUN = bootstrap_args.run_root.resolve()
RUNTIME_STATE = {"environment_created": False, "observed_frames": 0}
RUNTIME_ENV = None

def bootstrap_exception_hook(exc_type, exc, tb) -> None:
    traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
    try:
        write_implementation_exception(
            RUN / "implementation_exception.json", exc,
            {**RUNTIME_STATE, "exception_phase": "MODULE_BOOTSTRAP"},
        )
    finally:
        status = {
            "status": "INVALID", "primary_reason": "IMPLEMENTATION_EXCEPTION",
            "environment_created": RUNTIME_STATE["environment_created"],
            "observed_frames": RUNTIME_STATE["observed_frames"],
            "multiple_isaac_processes": False, "error": repr(exc),
        }
        RUN.mkdir(parents=True, exist_ok=True)
        status_tmp = RUN / "runner_status.json.tmp"
        status_tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        status_tmp.replace(RUN / "runner_status.json")
        sys.stdout.flush()
        sys.stderr.flush()
        for resource_name in ("RUNTIME_ENV", "simulation_app"):
            resource = globals().get(resource_name)
            if resource is not None:
                try:
                    resource.close()
                except BaseException:
                    traceback.print_exc(file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()

sys.excepthook = bootstrap_exception_hook

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

import carb
import torch
from PIL import Image
from omni.physx import get_physx_property_query_interface, get_physx_scene_query_interface
from omni.physx.bindings._physx import PhysxPropertyQueryMode, PhysxPropertyQueryResult
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
from isaaclab.envs import ManagerBasedEnv
from isaacsim.core.utils.stage import get_current_stage
from agile.rl_env.assets.robots.unitree_g1 import G1_W_HANDS_AGILE_ACTION_SCALE

from g1_access_push.sim.stage2.s2_01_box_env import (
    S2_01_BOX_SENSOR_CONFIGURED_PRIM_PATH,
    S2_01_ROBOT_FILTER_CONFIGURED_EXPRESSIONS,
    build_s2_01_env_cfg,
)
from g1_access_push.stage2.s2_01_contract import load_config, sha256_file, write_json

BOX_PRIM_PATH = "/World/envs/env_0/Box"
BOX_COLLIDER_PATH = BOX_PRIM_PATH + "/geometry/mesh"
BOX_MATERIAL_PATH = BOX_PRIM_PATH + "/geometry/material"


def process_count() -> int:
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore").lower()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "run_s2_01_box_stand_sanity.py" in cmd:
            count += 1
    return count


def config_instance_preflight(cfg, config: dict) -> dict:
    """Persist the resolved scene-instance contract before environment creation."""
    checkpoint = Path(cfg.actions.lower_body_joint_pos.policy_path).resolve()
    checkpoint_validation = validate_controller_checkpoint(
        checkpoint, None, config["robot"]["controller_checkpoint_sha256"],
        config["robot"]["forbidden_checkpoint_sha256s"],
        config["robot"]["forbidden_checkpoint_basenames"],
    )
    result = {
        "schema_version": 1,
        "status": "FAIL",
        "config_type": f"{type(cfg).__module__}.{type(cfg).__qualname__}",
        "scene_config_type": f"{type(cfg.scene).__module__}.{type(cfg.scene).__qualname__}",
        "config_is_instance": not isinstance(cfg, type) and not isinstance(cfg.scene, type),
        "terrain_present": hasattr(cfg.scene, "terrain") and cfg.scene.terrain is not None,
        "robot_present": hasattr(cfg.scene, "robot") and cfg.scene.robot is not None,
        "box_present": hasattr(cfg.scene, "box") and cfg.scene.box is not None,
        "num_envs": getattr(cfg.scene, "num_envs", None),
        "doorway_enabled": any("door" in name.lower() for name in vars(cfg.scene)),
        "nearby_obstacles_enabled": any("obstacle" in name.lower() for name in vars(cfg.scene)),
        "stage1_contract_sha": config["certification"]["standing_result_sha256"],
        "resolved_config_sha": sha256_file(args.resolved_config),
        "resolved_config_matches_pre_run_commit": sha256_file(args.resolved_config) == sha256_file(Path(__file__).resolve().parents[2] / "reports/stage2/s2_01_resolved_config.json"),
        **checkpoint_validation,
        "failure_reason": None,
    }
    failed = [name for name in ("config_is_instance", "terrain_present", "robot_present", "box_present") if not result[name]]
    if not result["resolved_config_matches_pre_run_commit"]:
        failed.append("resolved_config_matches_pre_run_commit")
    if result["num_envs"] != 1:
        failed.append("num_envs")
    if result["doorway_enabled"]:
        failed.append("doorway_enabled")
    if result["nearby_obstacles_enabled"]:
        failed.append("nearby_obstacles_enabled")
    if checkpoint_validation["reason"] == "FORBIDDEN_CHECKPOINT_SELECTED":
        failed.append("forbidden_checkpoint")
    elif checkpoint_validation["status"] != "PASS":
        failed.append("checkpoint_validation_failed")
    result["failure_reason"] = ",".join(failed) if failed else None
    result["status"] = "PASS" if not failed else "FAIL"
    write_json(RUN / "config_instance_preflight.json", result)
    if failed:
        raise RuntimeError(f"CONFIG_INSTANCE_PREFLIGHT_FAILED:{result['failure_reason']}")
    return result


def quaternion_rpy(q: torch.Tensor) -> tuple[float, float, float, float]:
    w, x, y, z = [float(item) for item in q]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    tilt = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))
    return roll, pitch, yaw, tilt


def wrapped_abs(value: float, reference: float) -> float:
    return abs(math.atan2(math.sin(value - reference), math.cos(value - reference)))


def physics_binding_record(stage: Usd.Stage, prim_path: str) -> dict:
    """Resolve a physics material without assuming a collider child name."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return {
            "prim_path": prim_path,
            "prim_valid": False,
            "direct_targets": [],
            "resolved_target": None,
            "relationship_path": None,
            "error": "INVALID_PRIM",
        }
    direct = prim.GetRelationship("material:binding:physics")
    direct_targets = [str(path) for path in direct.GetTargets()] if direct and direct.IsValid() else []
    resolved_target = direct_targets[0] if direct_targets else None
    relationship_path = str(direct.GetPath()) if direct and direct.IsValid() else None
    error = None
    try:
        material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        if material and material.GetPrim().IsValid():
            resolved_target = str(material.GetPath())
        if relationship and relationship.IsValid():
            relationship_path = str(relationship.GetPath())
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    return {
        "prim_path": prim_path,
        "prim_valid": True,
        "direct_targets": direct_targets,
        "resolved_target": resolved_target,
        "relationship_path": relationship_path,
        "error": error,
    }


def collision_binding_records(stage: Usd.Stage, root_path: str) -> list[dict]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return [physics_binding_record(stage, root_path)]
    return [
        physics_binding_record(stage, str(prim.GetPath()))
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def material_record(stage: Usd.Stage, path: str) -> dict:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return {"prim_valid": False, "prim_path": path}
    usd = UsdPhysics.MaterialAPI(prim)
    physx = PhysxSchema.PhysxMaterialAPI(prim)
    return {
        "prim_valid": True,
        "prim_path": path,
        "static_friction": float(usd.GetStaticFrictionAttr().Get()),
        "dynamic_friction": float(usd.GetDynamicFrictionAttr().Get()),
        "restitution": float(usd.GetRestitutionAttr().Get()),
        "friction_combine_mode": str(physx.GetFrictionCombineModeAttr().Get()),
        "restitution_combine_mode": str(physx.GetRestitutionCombineModeAttr().Get()),
    }


def query_mass_properties(stage: Usd.Stage, prim_path: str) -> dict:
    result: dict = {"done": False, "valid": False}
    stage_id = UsdUtils.StageCache().Get().GetId(stage).ToLongInt()
    prim_id = PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(prim_path).GetPath())

    def callback(info) -> None:
        result["done"] = True
        result["valid"] = info.result == PhysxPropertyQueryResult.VALID
        if result["valid"]:
            result.update({
                "mass": float(info.mass),
                "inertia": [float(value) for value in info.inertia],
                "principal_axes_wxyz": [float(info.principal_axes[3]), *[float(value) for value in info.principal_axes[:3]]],
                "center_of_mass": [float(value) for value in info.center_of_mass],
            })

    get_physx_property_query_interface().query_prim(
        stage_id=stage_id,
        prim_id=prim_id,
        query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS,
        rigid_body_fn=callback,
    )
    for _ in range(100):
        if result["done"]:
            break
        simulation_app.update()
    if not result["done"] or not result["valid"]:
        raise RuntimeError("MASS_PROPERTY_QUERY_FAILED")
    return result


def author_mass_properties(stage: Usd.Stage, config: dict) -> dict:
    prim = stage.GetPrimAtPath(BOX_PRIM_PATH)
    api = UsdPhysics.MassAPI(prim)
    if not api:
        api = UsdPhysics.MassAPI.Apply(prim)
    props = config["mass_properties"]
    mass = float(props["mass_kg"])
    com = [float(value) for value in props["center_of_mass_local_xyz_m"]]
    inertia = [float(value) for value in props["diagonal_inertia_kg_m2"]]
    axes = [float(value) for value in props["principal_axes_quaternion_wxyz"]]
    api.GetMassAttr().Set(mass)
    api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com))
    api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
    api.GetPrincipalAxesAttr().Set(Gf.Quatf(axes[0], Gf.Vec3f(*axes[1:])))
    return {
        "mass": float(api.GetMassAttr().Get()),
        "com": [float(value) for value in api.GetCenterOfMassAttr().Get()],
        "inertia": [float(value) for value in api.GetDiagonalInertiaAttr().Get()],
        "axes": [float(api.GetPrincipalAxesAttr().Get().GetReal()), *[float(value) for value in api.GetPrincipalAxesAttr().Get().GetImaginary()]],
    }


def set_and_read_runtime_mass_properties(box, config: dict) -> dict:
    """Set PhysX runtime values using getter-derived tensor shapes and read them back."""
    view = getattr(box, "root_physx_view", None)
    view_name = "root_physx_view"
    if view is None:
        view = getattr(box, "root_view", None)
        view_name = "root_view"
    if view is None:
        raise RuntimeError("RUNTIME_MASS_VIEW_UNAVAILABLE")
    required = ("get_masses", "set_masses", "get_coms", "set_coms", "get_inertias", "set_inertias")
    missing = [name for name in required if not callable(getattr(view, name, None))]
    if missing:
        raise RuntimeError(f"RUNTIME_MASS_VIEW_API_MISSING:{','.join(missing)}")

    masses = view.get_masses().clone()
    coms = view.get_coms().clone()
    inertias = view.get_inertias().clone()
    before = {
        "masses": masses.detach().cpu().tolist(),
        "coms_xyzw": coms.detach().cpu().tolist(),
        "inertias_column_major": inertias.detach().cpu().tolist(),
    }
    expected_mass = float(config["mass_properties"]["mass_kg"])
    expected_com = [float(value) for value in config["mass_properties"]["center_of_mass_local_xyz_m"]]
    expected_inertia = [float(value) for value in config["mass_properties"]["diagonal_inertia_kg_m2"]]
    masses[...] = expected_mass
    coms[..., :3] = torch.tensor(expected_com, dtype=coms.dtype, device=coms.device)
    coms[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=coms.dtype, device=coms.device)
    inertias.zero_()
    inertias[..., 0] = expected_inertia[0]
    inertias[..., 4] = expected_inertia[1]
    inertias[..., 8] = expected_inertia[2]
    indices = torch.arange(int(view.count), dtype=torch.int32, device="cpu")
    view.set_masses(masses, indices)
    view.set_inertias(inertias, indices)
    view.set_coms(coms, indices)

    read_masses = view.get_masses().clone()
    read_coms = view.get_coms().clone()
    read_inertias = view.get_inertias().clone()
    first_com = read_coms.reshape(-1, read_coms.shape[-1])[0]
    first_inertia = read_inertias.reshape(-1, read_inertias.shape[-1])[0]
    first_mass = read_masses.reshape(-1)[0]
    return {
        "runtime_view_attribute": view_name,
        "runtime_view_type": f"{type(view).__module__}.{type(view).__qualname__}",
        "getter_shapes": {
            "masses": list(read_masses.shape),
            "coms": list(read_coms.shape),
            "inertias": list(read_inertias.shape),
        },
        "quaternion_storage_order": "xyzw",
        "before": before,
        "mass": float(first_mass),
        "center_of_mass": [float(value) for value in first_com[:3]],
        "principal_axes_wxyz": [float(first_com[6]), *[float(value) for value in first_com[3:6]]],
        "inertia_matrix_column_major": [float(value) for value in first_inertia],
        "inertia": [float(first_inertia[index]) for index in (0, 4, 8)],
    }


def initial_robot_overlap_query(center_xyz: list[float], half_extent_xyz: list[float]) -> dict:
    hits: list[dict[str, str]] = []

    def callback(hit) -> bool:
        collision = str(getattr(hit, "collision", ""))
        rigid_body = str(getattr(hit, "rigid_body", ""))
        if collision.startswith("/World/envs/env_0/Robot") or rigid_body.startswith(
            "/World/envs/env_0/Robot"
        ):
            hits.append({"collision": collision, "rigid_body": rigid_body})
        return True

    total_hit_count = get_physx_scene_query_interface().overlap_box(
        carb.Float3(*half_extent_xyz),
        carb.Float3(*center_xyz),
        carb.Float4(0.0, 0.0, 0.0, 1.0),
        callback,
        False,
    )
    return {
        "query_backend": "CERTIFIED_PHYSX_SCENE_QUERY_OVERLAP_BOX",
        "all_scene_query_hit_count": int(total_hit_count),
        "robot_hit_count": len(hits),
        "robot_hits": hits,
    }


def main() -> None:
    global RUNTIME_ENV
    RUN.mkdir(parents=True, exist_ok=True)
    env = None
    run_mode = "preflight" if args.preflight_only else "formal"
    if args.preflight_only and not 2 <= args.preflight_steps <= 5:
        raise ValueError("preflight steps must be within [2, 5]")
    RUNTIME_STATE["mode"] = run_mode
    raw_status = {
        "status": "RUNNING", "primary_reason": None, "mode": run_mode,
        "multiple_isaac_processes": process_count() > 1,
        "observed_frames": 0,
        "environment_created": False,
    }
    write_json(RUN / "runner_status.json", raw_status)
    try:
        config = load_config(args.config)
        if raw_status["multiple_isaac_processes"]:
            raise RuntimeError("MULTIPLE_ISAAC_PROCESSES")
        standing_path = Path(config["certification"]["standing_result_path"])
        collision_path = Path(config["certification"]["collision_backend_path"])
        if sha256_file(standing_path) != config["certification"]["standing_result_sha256"] or json.loads(standing_path.read_text())["status"] != "PASS":
            raise RuntimeError("STANDING_ACTION_CONTRACT_UNCERTIFIED")
        if sha256_file(collision_path) != config["certification"]["collision_backend_sha256"] or not json.loads(collision_path.read_text())["passed"]:
            raise RuntimeError("COLLISION_BACKEND_UNCERTIFIED")
        cfg = build_s2_01_env_cfg()
        config_instance_preflight(cfg, config)
        cfg.seed = int(config["robot"]["fixed_seed"])
        cfg.sim.device = args.device
        env = ManagerBasedEnv(cfg=cfg)
        RUNTIME_ENV = env
        RUNTIME_STATE["environment_created"] = True
        raw_status["environment_created"] = True
        write_json(RUN / "runner_status.json", raw_status)
        stage = get_current_stage()
        env.reset(seed=cfg.seed)
        box = env.scene["box"]
        robot = env.scene["robot"]
        lower = env.action_manager.get_term("lower_body_joint_pos")
        camera = env.scene["audit_camera"]
        box_net = env.scene["box_net_contact"]
        box_robot = env.scene["box_robot_contact"]
        checkpoint = Path(lower.cfg.policy_path).resolve()
        runtime_checkpoint_validation = validate_controller_checkpoint(
            checkpoint, sha256_file(checkpoint), config["robot"]["controller_checkpoint_sha256"],
            config["robot"]["forbidden_checkpoint_sha256s"],
            config["robot"]["forbidden_checkpoint_basenames"],
        )
        if runtime_checkpoint_validation["status"] != "PASS":
            raise RuntimeError("STANDING_ACTION_CONTRACT_UNCERTIFIED")
        recurrent_reset = {
            "hidden_state_all_zero": bool(torch.count_nonzero(lower._policy.hidden_state).item() == 0),
            "cell_state_all_zero": bool(torch.count_nonzero(lower._policy.cell_state).item() == 0),
            "previous_policy_action_all_zero": bool(torch.count_nonzero(lower._previous_policy_actions).item() == 0),
            "last_policy_input_all_zero": bool(torch.count_nonzero(lower.last_policy_input).item() == 0),
        }
        if not all(recurrent_reset.values()):
            raise RuntimeError("STANDING_ACTION_CONTRACT_UNCERTIFIED")
        controller_audit = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "lower_body_joint_names": list(lower._joint_names),
            "lower_body_scale_source": "G1_W_HANDS_AGILE_ACTION_SCALE",
            "lower_body_scale_source_entry_count": len(G1_W_HANDS_AGILE_ACTION_SCALE),
            "resolved_lower_body_scale": [float(value) for value in lower._policy_output_scale[0]],
            "resolved_lower_body_offset": [float(value) for value in lower._policy_output_offset[0]],
            "lower_body_command": config["robot"]["lower_body_command"],
            "upper_body_action": "ZERO_DELTA_DEFAULT_ARMS_AND_WAIST",
            "recurrent_reset": recurrent_reset,
            "num_envs": cfg.scene.num_envs,
        }
        write_json(RUN / "controller_contract_audit.json", controller_audit)
        net_forces = box_net.data.net_forces_w
        robot_force_matrix = box_robot.data.force_matrix_w
        box_prim = stage.GetPrimAtPath(BOX_PRIM_PATH)
        contact_reporter_enabled = bool(box_prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
        robot_root_path = "/World/envs/env_0/Robot"
        robot_root = stage.GetPrimAtPath(robot_root_path)
        usd_candidate_robot_body_paths = [
            str(prim.GetPath()) for prim in Usd.PrimRange(robot_root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        net_force_available = net_forces is not None
        force_matrix_available = robot_force_matrix is not None
        net_force_finite = bool(net_force_available and torch.isfinite(net_forces).all())
        force_matrix_finite = bool(force_matrix_available and torch.isfinite(robot_force_matrix).all())
        backend_filter_count = int(box_robot.contact_physx_view.filter_count)
        net_body_audit = validate_resolved_sensor_body(
            S2_01_BOX_SENSOR_CONFIGURED_PRIM_PATH, box_net.cfg.prim_path,
            list(box_net.body_names), box_net.num_bodies,
            initialized=box_net.is_initialized,
            contact_reporter_enabled=contact_reporter_enabled,
            rigid_body_bound=bool(box_prim.HasAPI(UsdPhysics.RigidBodyAPI)),
            data_available=net_force_available,
        )
        robot_body_audit = validate_resolved_sensor_body(
            S2_01_BOX_SENSOR_CONFIGURED_PRIM_PATH, box_robot.cfg.prim_path,
            list(box_robot.body_names), box_robot.num_bodies,
            initialized=box_robot.is_initialized,
            contact_reporter_enabled=contact_reporter_enabled,
            rigid_body_bound=bool(box_prim.HasAPI(UsdPhysics.RigidBodyAPI)),
            data_available=force_matrix_available,
        )
        net_tensor_audit = validate_net_force_tensor(
            None if net_forces is None else list(net_forces.shape),
            num_envs=cfg.scene.num_envs, box_body_count=box_net.num_bodies,
            net_force_available=net_force_available, net_force_finite=net_force_finite,
        )
        filter_audit = validate_robot_filter_tensor(
            list(S2_01_ROBOT_FILTER_CONFIGURED_EXPRESSIONS),
            backend_filter_count,
            None if robot_force_matrix is None else list(robot_force_matrix.shape),
            num_envs=cfg.scene.num_envs, box_body_count=box_robot.num_bodies,
            usd_candidate_robot_rigid_body_count=len(usd_candidate_robot_body_paths),
            force_matrix_available=force_matrix_available,
            force_matrix_finite=force_matrix_finite,
        )
        initial_net_force_xyz = (
            [float(value) for value in net_forces[0, 0]] if net_force_available else None
        )
        contact_sensor_audit = {
            "box_net_sensor_prim_path": box_net.cfg.prim_path,
            "box_robot_sensor_prim_path": box_robot.cfg.prim_path,
            "box_net_body_names": list(box_net.body_names),
            "box_robot_body_names": list(box_robot.body_names),
            "box_robot_filter_paths": list(box_robot.cfg.filter_prim_paths_expr),
            "net_forces_available": net_force_available,
            "robot_force_matrix_available": force_matrix_available,
            "contact_reporter_initialized": bool(box_net.is_initialized and box_robot.is_initialized),
            "net_force_xyz_n": initial_net_force_xyz,
            "net_force_norm_n": (
                float(torch.linalg.vector_norm(net_forces[0, 0]).item()) if net_force_available else None
            ),
            "usd_candidate_robot_rigid_body_paths": usd_candidate_robot_body_paths,
            **net_body_audit,
            "robot_sensor_body_audit": robot_body_audit,
            **net_tensor_audit,
            **filter_audit,
        }
        contact_sensor_audit["sensor_audit_pass"] = bool(
            net_body_audit["sensor_body_audit_pass"]
            and robot_body_audit["sensor_body_audit_pass"]
            and net_tensor_audit["net_force_initialization_pass"]
            and filter_audit["filter_tensor_initialization_pass"]
            and filter_audit["configured_filter_pattern_count"] == 1
            and filter_audit["backend_filter_count"] == 1
        )
        write_json(RUN / "contact_sensor_audit.json", contact_sensor_audit)
        if not net_body_audit["path_audit_pass"] or not robot_body_audit["path_audit_pass"]:
            raise RuntimeError("CONTACT_SENSOR_AUDIT_IMPLEMENTATION_ERROR")
        if not contact_sensor_audit["sensor_audit_pass"]:
            raise RuntimeError("CONTACT_SENSOR_INITIALIZATION_FAILED")

        authored = author_mass_properties(stage, config)
        runtime = set_and_read_runtime_mass_properties(box, config)

        center = [float(value) for value in config["placement"]["box_spawn_center_xyz_m"]]
        rear = float(config["placement"]["box_rear_face_x_world_m"])
        cube = UsdGeom.Cube(stage.GetPrimAtPath(BOX_COLLIDER_PATH))
        cube_size = float(cube.GetSizeAttr().Get())
        cube_scale = cube.GetPrim().GetAttribute("xformOp:scale").Get() or Gf.Vec3d(1.0, 1.0, 1.0)
        rigid_api = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(BOX_PRIM_PATH))
        physx_rigid_api = PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(BOX_PRIM_PATH))
        rigid_audit = {
            "body_prim_path": BOX_PRIM_PATH,
            "cube_collider_prim_path": BOX_COLLIDER_PATH,
            "runtime_size_xyz_m": [cube_size * float(value) for value in cube_scale],
            "rigid_body_enabled": bool(rigid_api.GetRigidBodyEnabledAttr().Get()),
            "kinematic_enabled": bool(rigid_api.GetKinematicEnabledAttr().Get()),
            "gravity_enabled": not bool(physx_rigid_api.GetDisableGravityAttr().Get()),
        }
        write_json(RUN / "box_rigid_body_audit.json", rigid_audit)

        property_query = None
        property_query_error = None
        try:
            property_query = query_mass_properties(stage, BOX_PRIM_PATH)
        except Exception as exc:
            property_query_error = f"{type(exc).__name__}:{exc}"
        tolerance = float(config["mass_properties"]["mass_tolerance"])
        inertia_tol = 1.0e-5
        com_tol = 1.0e-5
        expected_inertia = config["mass_properties"]["diagonal_inertia_kg_m2"]
        expected_com = config["mass_properties"]["center_of_mass_local_xyz_m"]
        tolerance_checks = {
            "mass": abs(runtime["mass"] - float(config["mass_properties"]["mass_kg"])) <= tolerance,
            "com": max(abs(runtime["center_of_mass"][i] - expected_com[i]) for i in range(3)) <= com_tol,
            "inertia": max(abs(runtime["inertia"][i] - expected_inertia[i]) for i in range(3)) <= inertia_tol,
            "principal_axes": max(abs(runtime["principal_axes_wxyz"][i] - [1.0, 0.0, 0.0, 0.0][i]) for i in range(4)) <= com_tol,
        }
        property_query_disagreement = None
        if property_query is not None:
            property_query_disagreement = {
                "mass_abs_diff": abs(property_query["mass"] - runtime["mass"]),
                "com_max_abs_diff": max(
                    abs(property_query["center_of_mass"][i] - runtime["center_of_mass"][i])
                    for i in range(3)
                ),
                "inertia_max_abs_diff": max(
                    abs(property_query["inertia"][i] - runtime["inertia"][i])
                    for i in range(3)
                ),
            }
        mass_audit = {
            "body_prim_path": BOX_PRIM_PATH,
            "authoring_api": "UsdPhysics.MassAPI",
            "runtime_authority": runtime["runtime_view_attribute"],
            "runtime_view_type": runtime["runtime_view_type"],
            "runtime_getter_shapes": runtime["getter_shapes"],
            "runtime_quaternion_storage_order": runtime["quaternion_storage_order"],
            "runtime_values_before_set": runtime["before"],
            "authored_mass_kg": authored["mass"],
            "runtime_mass_kg": runtime["mass"],
            "authored_com_local_xyz_m": authored["com"],
            "runtime_com_local_xyz_m": runtime["center_of_mass"],
            "runtime_com_world_xyz_m": "PENDING_POST_SETTLE",
            "authored_diagonal_inertia_kg_m2": authored["inertia"],
            "runtime_diagonal_inertia_kg_m2": runtime["inertia"],
            "runtime_inertia_matrix_column_major": runtime["inertia_matrix_column_major"],
            "authored_principal_axes_wxyz": authored["axes"],
            "runtime_principal_axes_wxyz": runtime["principal_axes_wxyz"],
            "property_query": property_query,
            "property_query_error": property_query_error,
            "property_query_disagreement": property_query_disagreement,
            "property_query_is_authoritative": False,
            "tolerance_checks": tolerance_checks,
            "low_com_pass": tolerance_checks["com"],
        }
        write_json(RUN / "box_mass_properties_audit.json", mass_audit)

        terrain_root_path = str(cfg.scene.terrain.prim_path).rstrip("/")
        configured_ground_material_path = terrain_root_path + "/physicsMaterial"
        box_collision_bindings = collision_binding_records(stage, BOX_PRIM_PATH)
        ground_collision_bindings = collision_binding_records(stage, terrain_root_path)
        box_binding_targets = sorted({
            record["resolved_target"] for record in box_collision_bindings
            if record.get("resolved_target")
        })
        ground_binding_targets = sorted({
            record["resolved_target"] for record in ground_collision_bindings
            if record.get("resolved_target")
        })
        ground_material_path = (
            ground_binding_targets[0]
            if len(ground_binding_targets) == 1
            else configured_ground_material_path
        )
        box_material = material_record(stage, BOX_MATERIAL_PATH)
        ground_material = material_record(stage, ground_material_path)
        expected_material = config["materials"]["effective_pair"]
        material_keys = (
            "static_friction", "dynamic_friction", "restitution",
            "friction_combine_mode", "restitution_combine_mode",
        )
        box_material_values = {key: box_material.get(key) for key in material_keys}
        ground_material_values = {key: ground_material.get(key) for key in material_keys}
        pair_values_pass = (
            box_material.get("prim_valid") is True
            and ground_material.get("prim_valid") is True
            and box_material_values == expected_material
            and ground_material_values == expected_material
        )
        material_audit = {
            "box_material_prim_path": BOX_MATERIAL_PATH,
            "ground_material_prim_path": ground_material_path,
            "configured_ground_material_prim_path": configured_ground_material_path,
            "ground_material_path_resolution": (
                "UNIQUE_COLLISION_BINDING"
                if len(ground_binding_targets) == 1
                else "CONFIGURED_TERRAIN_ROOT_FALLBACK"
            ),
            "terrain_root_prim_path": terrain_root_path,
            "box_binding_target": box_binding_targets,
            "ground_binding_target": ground_binding_targets,
            "box_collision_bindings": box_collision_bindings,
            "ground_collision_bindings": ground_collision_bindings,
            "box": box_material_values,
            "ground": ground_material_values,
            "effective_pair": expected_material,
            "pair_pass": bool(
                pair_values_pass
                and BOX_MATERIAL_PATH in box_binding_targets
                and ground_material_path in ground_binding_targets
            ),
            "palm_box_material": {"status": "UNRESOLVED", "blocking_s2_01": False},
        }
        write_json(RUN / "physics_material_audit.json", material_audit)
        simulation_app.update()
        initial_overlap = initial_robot_overlap_query(
            center, [float(value) for value in config["geometry"]["half_extent_xyz_m"]]
        )
        initial_filter_norms = torch.linalg.vector_norm(robot_force_matrix[0, 0], dim=-1)
        initial_robot_contact_force_n = float(initial_filter_norms.amax().item())
        contact_sensor_audit.update({
            "initial_robot_filter_force_norms_n": [float(value) for value in initial_filter_norms],
            "initial_robot_contact_force_max_n": initial_robot_contact_force_n,
            "initial_robot_contact_pass": initial_robot_contact_force_n <= 0.0,
        })
        write_json(RUN / "contact_sensor_audit.json", contact_sensor_audit)
        geometry_audit = {
            "query_backend": initial_overlap["query_backend"],
            "collision_backend_calibration_sha256": config["certification"]["collision_backend_sha256"],
            "box_center_xyz_world_m": center,
            "box_rear_face_x_world_m": rear,
            "overlap_count": initial_overlap["robot_hit_count"],
            "scene_query_robot_hit": initial_overlap["robot_hit_count"] != 0,
            "scene_query_all_hit_count": initial_overlap["all_scene_query_hit_count"],
            "scene_query_robot_hits": initial_overlap["robot_hits"],
            "robot_collider_envelope": config["placement"]["robot_collider_envelope"],
            "nominal_base_to_box_distance": config["placement"]["nominal_base_to_box_distance"],
            "precontact_gap": config["placement"]["precontact_gap"],
            "robot_root_subtree_aabb": "OPTIONAL_METRIC_NOT_RUN",
            "doorway_count": 0,
            "obstacle_count": 0,
        }
        write_json(RUN / "scene_geometry_audit.json", geometry_audit)
        sim_dump = cfg.sim.to_dict()
        sim_text = json.dumps(sim_dump, sort_keys=True, default=str)
        write_json(RUN / "stage1_simulation_config_audit.json", {
            "resolved_simulation_cfg": sim_dump,
            "resolved_simulation_cfg_sha256": hashlib.sha256(sim_text.encode()).hexdigest(),
            "physics_frequency_hz": 1.0 / cfg.sim.dt,
            "control_frequency_hz": 1.0 / (cfg.sim.dt * cfg.decimation),
            "actor_override": "NOT_AUTHORED",
            "effective_source": "PHYSX_SCENE_OR_ENGINE_DEFAULT",
        })

        camera.set_world_poses_from_view(
            torch.tensor([[3.1, -3.0, 1.8]], device=env.device),
            torch.tensor([[0.8, 0.0, 0.65]], device=env.device),
        )
        names = env.action_manager.active_terms
        dims = env.action_manager.action_term_dim
        lower_index = names.index("lower_body_joint_pos")
        lower_start = sum(dims[:lower_index])
        lower_slice = slice(lower_start, lower_start + dims[lower_index])
        actions = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
        actions[:, lower_slice] = torch.tensor(config["robot"]["lower_body_command"], device=env.device)
        if not bool(torch.isfinite(actions).all()):
            raise RuntimeError("NONFINITE")

        expected_frames = (
            int(args.preflight_steps)
            if args.preflight_only
            else int(config["evaluation"]["expected_frames"])
        )
        reference_frame = (
            0 if args.preflight_only else int(config["settling"]["reference_trace_frame"])
        )
        reference_pos = None
        reference_yaw = None
        last_image = None
        episode_robot_contact_force_max_n = 0.0
        episode_robot_contact_force_sum_n_max = 0.0
        episode_robot_contact_nonzero_filter_count_max = 0
        with (RUN / "trace.jsonl").open("w", encoding="utf-8", buffering=1) as trace:
            for frame in range(expected_frames):
                env.step(actions)
                box_pos = box.data.root_link_pos_w[0].clone()
                box_quat = box.data.root_link_quat_w[0].clone()
                roll, pitch, yaw, _ = quaternion_rpy(box_quat)
                if frame == reference_frame:
                    reference_pos = box_pos.clone()
                    reference_yaw = yaw
                translation = float(torch.linalg.vector_norm(box_pos - reference_pos).item()) if reference_pos is not None else 0.0
                yaw_change = wrapped_abs(yaw, reference_yaw) if reference_yaw is not None else 0.0
                root_roll, root_pitch, _, root_tilt = quaternion_rpy(robot.data.root_quat_w[0])
                net_force_xyz = box_net.data.net_forces_w[0, 0]
                ground_force = float(torch.linalg.vector_norm(net_force_xyz).item())
                partner_force_norms = torch.linalg.vector_norm(
                    box_robot.data.force_matrix_w[0, 0], dim=-1
                )
                partner_force_norms_n = [float(value) for value in partner_force_norms]
                robot_force = max(partner_force_norms_n, default=0.0)
                robot_force_sum = sum(partner_force_norms_n)
                robot_nonzero_count = sum(value > 0.0 for value in partner_force_norms_n)
                episode_robot_contact_force_max_n = max(episode_robot_contact_force_max_n, robot_force)
                episode_robot_contact_force_sum_n_max = max(
                    episode_robot_contact_force_sum_n_max, robot_force_sum
                )
                episode_robot_contact_nonzero_filter_count_max = max(
                    episode_robot_contact_nonzero_filter_count_max, robot_nonzero_count
                )
                values = [
                    *[float(value) for value in box_pos], *[float(value) for value in box_quat],
                    float(torch.linalg.vector_norm(box.data.root_com_lin_vel_w[0]).item()),
                    float(torch.linalg.vector_norm(box.data.root_com_ang_vel_w[0]).item()),
                    float(robot.data.root_pos_w[0, 2]), root_roll, root_pitch, root_tilt,
                    ground_force, robot_force,
                ]
                controller_action_finite = bool(
                    torch.isfinite(lower._policy_actions).all()
                    and torch.isfinite(lower._processed_actions).all()
                )
                controller_observation_finite = bool(torch.isfinite(lower.last_policy_input).all())
                record = {
                    "frame": frame, "time_s": (frame + 1) * cfg.sim.dt * cfg.decimation,
                    "run_mode": run_mode,
                    "controller_action_finite": controller_action_finite,
                    "controller_observation_finite": controller_observation_finite,
                    "box_position_xyz_m": [float(value) for value in box_pos],
                    "box_quaternion_wxyz": [float(value) for value in box_quat],
                    "box_linear_speed_mps": values[7],
                    "box_angular_speed_radps": values[8],
                    "box_translation_from_reference_m": translation,
                    "box_abs_yaw_change_rad": yaw_change,
                    "box_abs_roll_rad": abs(roll), "box_abs_pitch_rad": abs(pitch),
                    "box_abs_vertical_drift_m": abs(float(box_pos[2] - reference_pos[2])) if reference_pos is not None else 0.0,
                    "box_ground_contact_force_n": ground_force,
                    "box_robot_contact_force_n": robot_force,
                    "box_net_normal_force_xyz_n": [float(value) for value in net_force_xyz],
                    "box_net_normal_force_norm_n": ground_force,
                    "robot_filter_force_norms_n": partner_force_norms_n,
                    "robot_contact_force_max_n": robot_force,
                    "robot_contact_force_sum_n": robot_force_sum,
                    "robot_contact_nonzero_filter_count": robot_nonzero_count,
                    "robot_box_contact": robot_force > 0.0,
                    "root_height_m": values[9], "root_roll_rad": root_roll,
                    "root_pitch_rad": root_pitch, "root_tilt_rad": root_tilt,
                    "robot_fall": values[9] < config["robot"]["fall_root_height_threshold_m"],
                    "robot_bad_tilt": root_tilt > config["robot"]["bad_tilt_threshold_rad"],
                    "post_initial_reset_count": 0,
                    "finite": bool(
                        all(math.isfinite(value) for value in values)
                        and controller_action_finite
                        and controller_observation_finite
                    ),
                }
                trace.write(json.dumps(record, sort_keys=True) + "\n")
                RUNTIME_STATE["observed_frames"] = frame + 1
                raw_status["observed_frames"] = frame + 1
                if args.preflight_only or frame % 100 == 99:
                    print(f"PHASE={run_mode} frame={frame + 1}/{expected_frames}", flush=True)
                rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
                last_image = Image.fromarray(rgb.astype("uint8"))
        contact_sensor_audit.update({
            "net_force_xyz_n": [float(value) for value in box_net.data.net_forces_w[0, 0]],
            "net_force_norm_n": float(torch.linalg.vector_norm(box_net.data.net_forces_w[0, 0]).item()),
            "episode_robot_contact_force_max_n": episode_robot_contact_force_max_n,
            "episode_robot_contact_force_sum_n_max": episode_robot_contact_force_sum_n_max,
            "episode_robot_contact_nonzero_filter_count_max": episode_robot_contact_nonzero_filter_count_max,
        })
        write_json(RUN / "contact_sensor_audit.json", contact_sensor_audit)
        if last_image is None:
            raise RuntimeError("MISSING_FINAL_IMAGE")
        image_path = RUN / ("preflight.png" if args.preflight_only else "final.png")
        last_image.save(image_path)
        mass_audit["runtime_com_world_xyz_m"] = [float(value) for value in box.data.root_com_pos_w[0]]
        mass_audit["com_height_above_ground_m"] = float(box.data.root_com_pos_w[0, 2])
        mass_audit["low_com_pass"] = bool(
            mass_audit["low_com_pass"]
            and mass_audit["com_height_above_ground_m"] < float(box.data.root_link_pos_w[0, 2])
            and abs(mass_audit["com_height_above_ground_m"] - 0.2) <= config["evaluation"]["stillness_thresholds"]["max_box_vertical_drift_m"]
        )
        write_json(RUN / "box_mass_properties_audit.json", mass_audit)
        if args.preflight_only:
            trace_lines = sum(
                1 for line in (RUN / "trace.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            preflight_checks = {
                "box_rigid_body": bool(
                    rigid_audit["rigid_body_enabled"]
                    and not rigid_audit["kinematic_enabled"]
                    and rigid_audit["gravity_enabled"]
                ),
                "runtime_mass_com_inertia": bool(
                    all(mass_audit["tolerance_checks"].values())
                    and mass_audit["low_com_pass"]
                ),
                "physics_material_binding": bool(material_audit["pair_pass"]),
                "contact_sensor": bool(contact_sensor_audit["sensor_audit_pass"]),
                "initial_robot_box_contact": bool(
                    contact_sensor_audit["initial_robot_contact_pass"]
                ),
                "initial_robot_box_overlap": bool(
                    geometry_audit["overlap_count"] == 0
                    and not geometry_audit["scene_query_robot_hit"]
                ),
                "controller_action_finite": bool(
                    torch.isfinite(lower._policy_actions).all()
                    and torch.isfinite(lower._processed_actions).all()
                ),
                "controller_observation_finite": bool(
                    torch.isfinite(lower.last_policy_input).all()
                ),
                "trace_writer": trace_lines == expected_frames,
                "camera_output": image_path.is_file() and image_path.stat().st_size > 0,
                "no_second_isaac": not raw_status["multiple_isaac_processes"],
            }
            failed_checks = [
                name for name, passed in preflight_checks.items() if not passed
            ]
            preflight_result = {
                "schema_version": 1,
                "stage": "S2-01",
                "mode": "PREFLIGHT_ONLY",
                "status": "PASS" if not failed_checks else "FAIL",
                "primary_reason": (
                    "ALL_PREFLIGHT_GATES_PASSED"
                    if not failed_checks
                    else "PREFLIGHT_GATES_FAILED"
                ),
                "failed_checks": failed_checks,
                "checks": preflight_checks,
                "steps": expected_frames,
                "config_sha256": sha256_file(args.config),
                "resolved_config_sha256": sha256_file(args.resolved_config),
                "scientific_result_created": False,
            }
            write_json(RUN / "preflight_result.json", preflight_result)
            if failed_checks:
                raw_status.update({
                    "status": "PREFLIGHT_FAILED",
                    "primary_reason": "PREFLIGHT_GATES_FAILED",
                    "failed_checks": failed_checks,
                    "observed_frames": expected_frames,
                })
                write_json(RUN / "runner_status.json", raw_status)
                return
        raw_status.update({
            "status": "COMPLETE",
            "primary_reason": None,
            "observed_frames": expected_frames,
        })
        write_json(RUN / "runner_status.json", raw_status)
    except Exception as exc:
        raw_status.update({"status": "INVALID", "primary_reason": "IMPLEMENTATION_EXCEPTION", "error": repr(exc)})
        write_json(RUN / "runner_status.json", raw_status)
        raise


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        write_implementation_exception(
            RUN / "implementation_exception.json", exc,
            {**RUNTIME_STATE, "exception_phase": "MAIN"},
        )
        exit_code = 1
    try:
        if RUNTIME_ENV is not None:
            RUNTIME_ENV.close()
        simulation_app.close()
    except BaseException as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        if exit_code == 0:
            write_implementation_exception(
                RUN / "implementation_exception.json", exc,
                {**RUNTIME_STATE, "exception_phase": "CLEANUP"},
            )
            exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(exit_code)
