"""Isaac/PhysX runtime instrumentation for S2-03T path recovery.

Import this module only after :class:`isaaclab.app.AppLauncher` has started.
It contains no policy or controller changes; it only resolves colliders,
constructs one-to-one pelvis contact queries, and records evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from isaaclab.sensors import ContactSensor, ContactSensorCfg


CONTACT_EPS_N = 1.0e-6


def _json_safe(value: Any) -> Any:
    """Normalize tensors/NumPy values and unresolved non-finite diagnostics."""
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else "METRIC_MISSING"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tolist(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): tolist(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tolist(v) for v in value]
    return value


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = list(values)
    return float(np.percentile(np.asarray(vals, dtype=float), q)) if vals else None


def root_metrics(quat: torch.Tensor) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))
    return roll, pitch, yaw, tilt


def term_joint_ids(term: Any, count: int) -> list[int]:
    ids = getattr(term, "_joint_ids", None)
    if isinstance(ids, slice):
        return [int(v) for v in list(range(count))[ids]]
    return [] if ids is None else [int(v) for v in ids]


def path_template(path: str) -> str:
    return path.replace("/World/envs/env_0/", "/World/envs/env_.*/")


def owning_body_name(prim: Any, body_names: set[str]) -> str | None:
    current = prim
    while current and current.IsValid():
        name = str(current.GetName())
        if name in body_names:
            return name
        current = current.GetParent()
    return None


def body_path_map(stage: Any) -> dict[str, str]:
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    result: dict[str, str] = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            result.setdefault(str(prim.GetName()), str(prim.GetPath()))
    return result


def _matrix(value: Gf.Matrix4d) -> list[list[float]]:
    return [[(float(value[i][j]) if math.isfinite(float(value[i][j])) else "METRIC_MISSING") for j in range(4)] for i in range(4)]


def collider_audit(stage: Any, robot: Any, body_paths: dict[str, str]) -> dict[str, Any]:
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    body_names = set(str(v) for v in robot.body_names)
    entries: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        is_collision = prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(PhysxSchema.PhysxCollisionAPI) or "/collisions/" in path.lower()
        if not is_collision:
            continue
        owner = owning_body_name(prim, body_names)
        if owner is None:
            continue
        xform = UsdGeom.Xformable(prim)
        local = xform.GetLocalTransformation() if xform else Gf.Matrix4d(1.0)
        world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()) if xform else Gf.Matrix4d(1.0)
        collision = UsdPhysics.CollisionAPI(prim)
        enabled_attr = collision.GetCollisionEnabledAttr()
        enabled = enabled_attr.Get() if enabled_attr.IsValid() else None
        mesh = UsdPhysics.MeshCollisionAPI(prim)
        approximation_attr = mesh.GetApproximationAttr()
        approximation = approximation_attr.Get() if approximation_attr.IsValid() else None
        contact_offset: Any = "METRIC_MISSING"
        rest_offset: Any = "METRIC_MISSING"
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physics = PhysxSchema.PhysxCollisionAPI(prim)
            contact_attr, rest_attr = physics.GetContactOffsetAttr(), physics.GetRestOffsetAttr()
            contact_offset = contact_attr.Get() if contact_attr.IsValid() else "METRIC_MISSING"
            rest_offset = rest_attr.Get() if rest_attr.IsValid() else "METRIC_MISSING"
        scale = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractRotationMatrix()
        entries.append(
            {
                "collider_prim": str(prim.GetPath()),
                "owning_rigid_body": owner,
                "owning_body_path": body_paths.get(owner, "UNKNOWN"),
                "visual_mesh_prim": str(prim.GetPath()) if prim.GetTypeName() == "Mesh" else "METRIC_MISSING",
                "type_name": prim.GetTypeName(),
                "collider_approximation": str(approximation) if approximation is not None else "DEFAULT_OR_NOT_APPLICABLE",
                "local_transform": _matrix(local),
                "world_transform_at_audit": _matrix(world),
                "scale_rotation_matrix_product": ([[float(scale[i][j]) for j in range(3)] for i in range(3)] if all(math.isfinite(float(scale[i][j])) for i in range(3) for j in range(3)) else "METRIC_MISSING"),
                "contact_offset_m": contact_offset,
                "rest_offset_m": rest_offset,
                "collision_enabled": enabled,
                "filtered_pair_state": "NO_COLLISION_FILTER_MODIFICATION_APPLIED",
            }
        )
    by_body: dict[str, list[str]] = {}
    for item in entries:
        by_body.setdefault(item["owning_rigid_body"], []).append(item["collider_prim"])
    return {"status": "PASS" if entries and "pelvis" in by_body else "INVALID", "count": len(entries),
            "entries": entries, "colliders_by_body": by_body}


def body_state(robot: Any, body_name: str | None) -> dict[str, Any] | str:
    if body_name is None or body_name not in robot.body_names:
        return "METRIC_MISSING"
    index = list(robot.body_names).index(body_name)
    velocity = robot.data.body_vel_w[0, index] if hasattr(robot.data, "body_vel_w") else None
    return {
        "body_id": index,
        "position_world_m": tolist(robot.data.body_pos_w[0, index]),
        "quaternion_world_wxyz": tolist(robot.data.body_quat_w[0, index]),
        "linear_velocity_world_mps": "METRIC_MISSING" if velocity is None else tolist(velocity[:3]),
        "angular_velocity_world_rad_s": "METRIC_MISSING" if velocity is None else tolist(velocity[3:]),
    }


def group_for_body(owner: str) -> str:
    value = owner.lower()
    if owner == "ground":
        return "ground"
    if "hand_" in value and "palm" not in value or "finger" in value:
        return "fingers"
    if value.startswith("left_") and any(token in value for token in ("shoulder", "elbow", "wrist", "hand", "arm")):
        return "left_arm"
    if value.startswith("right_") and any(token in value for token in ("shoulder", "elbow", "wrist", "hand", "arm")):
        return "right_arm"
    if any(token in value for token in ("torso", "waist", "head")):
        return "torso"
    return "other"


class ContactPairAuditor:
    """Resolve exact pelvis-versus-rigid-body pairs with one-to-one filters."""

    def __init__(self, env: Any, stage: Any, runtime: dict[str, Any], colliders: dict[str, Any], dt_s: float) -> None:
        self.env, self.stage, self.robot, self.dt_s = env, stage, env.scene["robot"], dt_s
        self.colliders = colliders
        self.pelvis_path = runtime["body_paths"].get("pelvis")
        if not self.pelvis_path:
            raise RuntimeError("PELVIS_RIGID_BODY_PATH_MISSING")
        candidates: list[dict[str, str]] = []
        for owner, paths in colliders["colliders_by_body"].items():
            if owner == "pelvis":
                continue
            body_path = runtime["body_paths"].get(owner)
            if body_path:
                candidates.append({"owner": owner, "body_path": body_path,
                                   "filter_path": body_path, "colliders": list(paths)})
        world = stage.GetPrimAtPath("/World")
        ground_paths: list[str] = []
        for prim in Usd.PrimRange(world, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if path.startswith("/World/envs/env_0/Robot"):
                continue
            if "ground" in path.lower() and prim.HasAPI(UsdPhysics.CollisionAPI):
                ground_paths.append(path)
        for path in sorted(set(ground_paths)):
            candidates.append({"owner": "ground", "body_path": path, "filter_path": path, "colliders": [path]})

        self.entries: list[dict[str, Any]] = []
        self.initialization_errors: list[dict[str, str]] = []
        self.history: dict[str, list[dict[str, Any]]] = {}
        initialized_groups: set[str] = set()
        for candidate in candidates:
            cfg = ContactSensorCfg(
                prim_path=path_template(self.pelvis_path),
                update_period=0.0,
                history_length=0,
                track_pose=True,
                track_contact_points=True,
                max_contact_data_count_per_prim=64,
                filter_prim_paths_expr=[path_template(candidate["filter_path"])],
                debug_vis=False,
            )
            try:
                sensor = ContactSensor(cfg)
                sensor._initialize_impl()
                sensor._is_initialized = True
                if sensor.num_bodies != 1 or int(sensor.contact_physx_view.filter_count) != 1:
                    raise RuntimeError(f"PAIR_SENSOR_SHAPE_INVALID:{sensor.num_bodies}:{sensor.contact_physx_view.filter_count}")
                entry = {**candidate, "sensor": sensor, "filter_template": cfg.filter_prim_paths_expr[0]}
                self.entries.append(entry)
                self.history[candidate["body_path"]] = []
                initialized_groups.add(group_for_body(candidate["owner"]))
            except Exception as exc:
                self.initialization_errors.append({"owner": candidate["owner"], "filter_path": candidate["filter_path"],
                                                   "error": f"{type(exc).__name__}:{exc}"})
        required = {"ground", "left_arm", "right_arm", "torso", "fingers"}
        self.coverage = {
            "required_groups": sorted(required),
            "initialized_groups": sorted(initialized_groups),
            "missing_groups": sorted(required - initialized_groups),
            "candidate_count": len(candidates),
            "initialized_count": len(self.entries),
            "initialization_errors": self.initialization_errors,
        }
        self.coverage["status"] = "PASS" if self.entries and not self.coverage["missing_groups"] else "INVALID"

    def reset(self) -> None:
        for entry in self.entries:
            entry["sensor"].reset()

    def aggregate_pelvis_force(self) -> float | None:
        sensor = self.env.scene["contact_forces"]
        forces, names = sensor.data.net_forces_w, list(sensor.body_names)
        if forces is None or "pelvis" not in names:
            return None
        value = float(torch.linalg.vector_norm(forces[0, names.index("pelvis")]))
        return value if math.isfinite(value) else None

    def _raw_contact(self, sensor: ContactSensor) -> dict[str, Any]:
        result: dict[str, Any] = {"normal_force_scalar_n": "METRIC_MISSING",
                                  "contact_point_world_xyz_m": "METRIC_MISSING",
                                  "contact_normal_world_xyz": "METRIC_MISSING",
                                  "separation_m": "METRIC_MISSING", "raw_contact_count": 0}
        try:
            forces, points, normals, distances, counts, starts = sensor.contact_physx_view.get_contact_data(
                dt=self.env.sim.get_physics_dt())
            counts_np = np.asarray(counts.detach().cpu() if isinstance(counts, torch.Tensor) else counts).reshape(-1)
            starts_np = np.asarray(starts.detach().cpu() if isinstance(starts, torch.Tensor) else starts).reshape(-1)
            count = int(counts_np[0]) if len(counts_np) else 0
            start = int(starts_np[0]) if len(starts_np) else 0
            result["raw_contact_count"] = count
            if count <= 0:
                return result
            arrays = []
            for value in (forces, points, normals, distances):
                arrays.append(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
            force_np, point_np, normal_np, distance_np = arrays
            force_np = force_np.reshape(-1)
            point_np, normal_np, distance_np = point_np.reshape(-1, 3), normal_np.reshape(-1, 3), distance_np.reshape(-1)
            stop = min(start + count, len(point_np), len(normal_np))
            indices = list(range(start, stop))
            if not indices:
                return result
            force_slice = force_np[start:min(start + count, len(force_np))]
            index = indices[int(np.argmax(np.abs(force_slice)))] if len(force_slice) == len(indices) else indices[0]
            result.update(
                {
                    "normal_force_scalar_n": float(force_np[index]) if index < len(force_np) else float(np.max(np.abs(force_slice))),
                    "contact_point_world_xyz_m": point_np[index].astype(float).tolist(),
                    "contact_normal_world_xyz": normal_np[index].astype(float).tolist(),
                    "separation_m": float(distance_np[index]) if index < len(distance_np) else "METRIC_MISSING",
                }
            )
        except Exception as exc:
            result["raw_contact_error"] = f"{type(exc).__name__}:{exc}"
        return result

    def _pelvis_colliders(self) -> list[str]:
        return list(self.colliders["colliders_by_body"].get("pelvis", []))

    def sample(self, frame: int, phase: str, arm_q: torch.Tensor, *, force_all: bool = False,
               record_history: bool = True) -> dict[str, Any]:
        aggregate = self.aggregate_pelvis_force()
        contacts: list[dict[str, Any]] = []
        if force_all or aggregate is None or aggregate > CONTACT_EPS_N:
            for entry in self.entries:
                sensor: ContactSensor = entry["sensor"]
                sensor.update(self.dt_s, force_recompute=True)
                matrix = sensor.data.force_matrix_w
                if matrix is None or list(matrix.shape) != [1, 1, 1, 3]:
                    raise RuntimeError(f"PAIR_SENSOR_FORCE_SHAPE_INVALID:{entry['body_path']}:{None if matrix is None else list(matrix.shape)}")
                vector = matrix[0, 0, 0].detach().clone()
                force_norm = float(torch.linalg.vector_norm(vector))
                raw = self._raw_contact(sensor)
                in_contact = force_norm > CONTACT_EPS_N or int(raw.get("raw_contact_count", 0)) > 0
                if not in_contact and not force_all:
                    continue
                state_a, state_b = body_state(self.robot, "pelvis"), body_state(self.robot, entry["owner"] if entry["owner"] != "ground" else None)
                relative: Any = "METRIC_MISSING"
                if isinstance(state_a, dict) and isinstance(state_b, dict):
                    va, vb = state_a["linear_velocity_world_mps"], state_b["linear_velocity_world_mps"]
                    if isinstance(va, list) and isinstance(vb, list):
                        relative = [float(b) - float(a) for a, b in zip(va, vb, strict=True)]
                root_pose = torch.cat((self.robot.data.root_link_pos_w[0], self.robot.data.root_link_quat_w[0]))
                record = {
                    "phase": phase, "frame": int(frame), "time_s": float(frame * self.dt_s),
                    "body_a": self.pelvis_path, "body_b": entry["body_path"],
                    "body_a_name": "pelvis", "body_b_name": entry["owner"],
                    "candidate_colliders_a": self._pelvis_colliders(), "candidate_colliders_b": entry["colliders"],
                    "collider_pair_resolution": "EXACT" if len(self._pelvis_colliders()) == len(entry["colliders"]) == 1 else "BODY_EXACT_COLLIDER_SET_RECORDED",
                    "normal_force_vector_n": tolist(vector), "force_norm_n": force_norm, **raw,
                    "relative_body_linear_velocity_world_mps": relative,
                    "body_pose_a": state_a, "body_pose_b": state_b,
                    "arm_joint_positions_rad": tolist(arm_q), "root_pose_world_xyz_wxyz": tolist(root_pose),
                    "post_step_measurement": phase not in {"STATIC", "FINAL_STATIC_TARGET", "DEFAULT_STATIC"},
                }
                if in_contact:
                    contacts.append(record)
                    if record_history:
                        self.history[entry["body_path"]].append(record)
        best = max(contacts, key=lambda item: float(item["force_norm_n"]), default=None)
        return {"aggregate_pelvis_force_n": aggregate, "contact_count": len(contacts), "contacts": contacts,
                "pair_force_norm_n": 0.0 if best is None else best["force_norm_n"],
                "closest_pair": "NONE" if best is None else f"pelvis::{best['body_b_name']}", "best": best}

    def summaries(self, default_pairs: set[str], final_pairs: set[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for body_path, records in self.history.items():
            if not records:
                continue
            first, last = records[0], records[-1]
            maximum = max(records, key=lambda item: float(item["force_norm_n"]))
            numeric_separations = [float(r["separation_m"]) for r in records if isinstance(r["separation_m"], (int, float))]
            output.append(
                {
                    "body_a": first["body_a"], "body_b": first["body_b"],
                    "body_a_name": first["body_a_name"], "body_b_name": first["body_b_name"],
                    "candidate_colliders_a": first["candidate_colliders_a"], "candidate_colliders_b": first["candidate_colliders_b"],
                    "collider_pair_resolution": first["collider_pair_resolution"],
                    "first_contact_frame": first["frame"], "first_contact_time_s": first["time_s"],
                    "last_contact_frame": last["frame"], "last_contact_time_s": last["time_s"],
                    "contact_duration_s": (last["frame"] - first["frame"] + 1) * self.dt_s,
                    "force_max_n": maximum["force_norm_n"], "force_vector_at_max_n": maximum["normal_force_vector_n"],
                    "contact_point_world_xyz_m": maximum["contact_point_world_xyz_m"],
                    "contact_normal_world_xyz": maximum["contact_normal_world_xyz"],
                    "minimum_separation_m": min(numeric_separations) if numeric_separations else "METRIC_MISSING",
                    "relative_body_velocity_at_first_mps": first["relative_body_linear_velocity_world_mps"],
                    "body_pose_a_at_first": first["body_pose_a"], "body_pose_b_at_first": first["body_pose_b"],
                    "arm_joint_positions_at_first_rad": first["arm_joint_positions_rad"],
                    "root_pose_at_first_world_xyz_wxyz": first["root_pose_world_xyz_wxyz"],
                    "present_in_default_stand": body_path in default_pairs,
                    "present_at_final_static_target": body_path in final_pairs,
                    "observations": records,
                }
            )
        return sorted(output, key=lambda item: (item["first_contact_frame"], -float(item["force_max_n"])))


class BodyClearanceModel:
    """Conservative world collider-bound sphere clearance for dense replay."""

    def __init__(self, stage: Any, robot: Any, colliders: dict[str, Any]) -> None:
        self.robot, self.radii = robot, {}
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
        body_names = set(str(v) for v in robot.body_names)
        for item in colliders["entries"]:
            owner = str(item["owning_rigid_body"])
            if owner not in body_names:
                continue
            try:
                box = cache.ComputeWorldBound(stage.GetPrimAtPath(item["collider_prim"])).ComputeAlignedRange()
                lo, hi = box.GetMin(), box.GetMax()
                body_id = list(robot.body_names).index(owner)
                center = robot.data.body_pos_w[0, body_id].detach().cpu().numpy()
                corners = np.asarray([[x, y, z] for x in (float(lo[0]), float(hi[0]))
                                      for y in (float(lo[1]), float(hi[1])) for z in (float(lo[2]), float(hi[2]))])
                radius = float(np.linalg.norm(corners - center.reshape(1, 3), axis=1).max())
                if math.isfinite(radius) and 0.0 < radius < 2.0:
                    self.radii[owner] = max(self.radii.get(owner, 0.0), radius)
            except Exception:
                continue
        self.pelvis_id = list(robot.body_names).index("pelvis")
        self.side_names = {
            side: [name for name in robot.body_names if name.startswith(f"{side}_") and any(
                token in name for token in ("shoulder", "elbow", "wrist", "hand", "arm"))]
            for side in ("left", "right")
        }

    def clearance(self, side: str) -> dict[str, Any]:
        pelvis_radius = self.radii.get("pelvis")
        if pelvis_radius is None:
            return {"clearance_m": None, "pair": "UNKNOWN", "method": "COLLIDER_BOUND_SPHERE_UNAVAILABLE"}
        pelvis = self.robot.data.body_pos_w[0, self.pelvis_id]
        values: list[tuple[float, str]] = []
        for name in self.side_names[side]:
            radius = self.radii.get(name)
            if radius is None:
                continue
            body_id = list(self.robot.body_names).index(name)
            distance = float(torch.linalg.vector_norm(self.robot.data.body_pos_w[0, body_id] - pelvis))
            values.append((distance - pelvis_radius - radius, name))
        if not values:
            return {"clearance_m": None, "pair": "UNKNOWN", "method": "COLLIDER_BOUND_SPHERE_UNAVAILABLE"}
        clearance, name = min(values)
        return {"clearance_m": clearance, "pair": f"pelvis::{name}",
                "method": "CONSERVATIVE_WORLD_COLLIDER_BOUNDING_SPHERES"}


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int = 25) -> None:
        self.path, self.frames = path, 0
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v",
             f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset",
             "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("VIDEO_STDIN_CLOSED")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> dict[str, Any]:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
            self.proc.stdin = None
        stderr = b"" if self.proc.stderr is None else self.proc.stderr.read()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"FFMPEG_FAILED:{rc}:{stderr.decode(errors='replace')[-500:]}")
        if not self.path.is_file() or self.path.stat().st_size <= 0:
            raise RuntimeError(f"VIDEO_EMPTY:{self.path}")
        return {"path": str(self.path), "size_bytes": self.path.stat().st_size,
                "sha256": sha256(self.path), "encoded_frames": self.frames}


def camera_rgb(camera: Any) -> np.ndarray:
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    return rgb.astype(np.uint8, copy=False)


class EvidenceVideo:
    def __init__(self, run_root: Path, env: Any, front_name: str, side_name: str) -> None:
        self.run_root, self.env = run_root, env
        self.front_camera, self.side_camera = env.scene["safe_chest_front_camera"], env.scene["safe_chest_side_camera"]
        self.width, self.height = int(self.front_camera.cfg.width), int(self.front_camera.cfg.height)
        self.front, self.side = VideoWriter(run_root / front_name, self.width, self.height), VideoWriter(run_root / side_name, self.width, self.height)
        self.font, self.counter = ImageFont.load_default(), 0

    def capture(self, record: dict[str, Any], *, force: bool = False, keyframe: str | None = None) -> None:
        lines = [
            f"phase={record.get('phase')} segment={record.get('segment','-')} frame={record.get('frame')}",
            f"q_clear/q_lift/q_chest={record.get('waypoint_symbols','-')}",
            f"closest_pair={record.get('closest_pair','UNKNOWN')}",
            f"pelvis_clearance={record.get('left_pelvis_clearance_m','UNKNOWN')}",
            f"force={record.get('pair_force_norm_n','UNKNOWN')} margin={record.get('minimum_joint_limit_margin_rad','UNKNOWN')}",
            f"torque={record.get('arm_torque_ratio_max','UNKNOWN')} root_tilt={record.get('root_tilt_deg','UNKNOWN')}",
            f"hold={record.get('hold_timer_s',0.0)} oscillation={record.get('oscillation_detected',False)}",
        ]

        def annotate(rgb: np.ndarray) -> Image.Image:
            image = Image.fromarray(rgb, mode="RGB")
            draw = ImageDraw.Draw(image, mode="RGBA")
            height = 13
            draw.rectangle((2, 2, 620, 8 + height * len(lines)), fill=(0, 0, 0, 180))
            for index, line in enumerate(lines):
                draw.text((6, 5 + index * height), line[:120], fill=(255, 255, 255, 255), font=self.font)
            return image

        front, side = annotate(camera_rgb(self.front_camera)), annotate(camera_rgb(self.side_camera))
        if force or self.counter % 2 == 0:
            self.front.write(np.asarray(front, dtype=np.uint8))
            self.side.write(np.asarray(side, dtype=np.uint8))
        if keyframe:
            front.save(self.run_root / f"frame_{keyframe}.png")
            side.save(self.run_root / f"frame_{keyframe}_side.png")
        self.counter += 1

    def close(self) -> dict[str, Any]:
        return {"front": self.front.close(), "side": self.side.close()}
