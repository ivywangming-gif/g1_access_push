#!/usr/bin/env python3
"""S2-03T safe chest joint-reference recovery qualification.

The campaign is intentionally a single no-box, one-environment Isaac run:
old-trace audit -> target-frame audit -> default-arm stand baseline -> static
candidate audit -> left/right/bilateral probes -> formal joint-space move and
10-second hold.  The formal arm commands are absolute joint targets; no
Differential IK term is instantiated in this environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np


BOOT = argparse.ArgumentParser(add_help=False)
BOOT.add_argument("--run-root", type=Path, required=True)
BOOT.add_argument("--reference", type=Path)
boot_args, _ = BOOT.parse_known_args()
RUN = boot_args.run_root.resolve()
RUN.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


write_json(
    RUN / "runner_status.json",
    {"status": "STARTING", "phase": "APP_LAUNCH", "stage": "S2-03T_SAFE_CHEST_JOINT_REFERENCE_RECOVERY"},
)

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--reference", type=Path)
parser.add_argument("--seed", type=int, default=42)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seed != 42:
    raise SystemExit("SAFE_CHEST_REFERENCE_SEED_MUST_BE_42")
simulation_app = AppLauncher(args).app

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Usd  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

from g1_access_push.sim.stage2.s2_03t_safe_chest_joint_reference_env_cfg import (  # noqa: E402
    S203TSafeChestJointReferenceEnvCfg,
)
from g1_access_push.stage2.s2_03t_safe_chest_joint_reference_contract import (  # noqa: E402
    ARM_JOINT_NAMES,
    CONTROL_DT_S,
    FORBIDDEN_FORCE_GATE_N,
    FORMAL_HOLD_SECONDS,
    FORMAL_HOLD_STEPS,
    JOINT_MARGIN_GATE_RAD,
    JOINT_RATE_LIMIT_RAD_S,
    LEFT_ARM_JOINT_NAMES,
    MOVE_STEPS,
    OLD_GEOMETRY_AUDIT_PATH,
    OLD_TRACE_PATH,
    PALM_BODY_NAMES,
    PALM_FRAME_NAMES,
    REFERENCE_PATH,
    REFERENCE_SHA256,
    RIGHT_ARM_JOINT_NAMES,
    ROOT_HEIGHT_RANGE_M,
    ROOT_TILT_GATE_DEG,
    S2_02_FORMAL_RUN,
    S2_02_PALM_QUATERNION_OBJECT_WXYZ,
    S2_02_PALM_LOCAL_NORMAL_AXIS,
    S2_02_PELVIS_SYMBOL,
    S2_02_SOURCE_SYMBOL,
    SETTLE_STEPS,
    SHORT_HOLD_STEPS,
    STATIC_MARGIN_GATE_RAD,
    STAGE,
    audit_old_trace,
    certified_object_local_targets,
    finite_number,
    minimum_jerk_fraction,
)


LOWER_BODY_COMMAND = (0.0, 0.0, 0.0, 0.7)
LOWER_CHECKPOINT_SHA256 = "a0151975a757a33f0f5ed236d5616e2a98643abe2b36e50ac9ad01b6dd1f6d7e"
ENV: Any = None
VIDEO_RECORDER: "EvidenceVideo | None" = None


def finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def tolist(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): tolist(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tolist(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


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
    if ids is None:
        return []
    return [int(v) for v in ids]


def forbidden_contact(sensor: Any) -> tuple[bool | None, float | None, list[str]]:
    forces = getattr(getattr(sensor, "data", None), "net_forces_w", None)
    if forces is None:
        return None, None, []
    values = forces[0]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = torch.linalg.vector_norm(values, dim=-1)
    if not bool(torch.isfinite(norms).all()):
        return None, None, []
    names = list(getattr(sensor, "body_names", ()))
    if len(names) != int(norms.numel()):
        return None, None, names
    bad: list[str] = []
    maximum = 0.0
    for name, force in zip(names, norms, strict=True):
        if "ankle" in name or "foot" in name:
            continue
        value = float(force)
        maximum = max(maximum, value)
        if value > FORBIDDEN_FORCE_GATE_N:
            bad.append(name)
    return bool(bad), maximum, bad


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int = 25) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.path = path
        self.frames = 0
        self.proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("VIDEO_STDIN_CLOSED")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
            self.proc.stdin = None
        stderr = b"" if self.proc.stderr is None else self.proc.stderr.read()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"FFMPEG_FAILED:{rc}:{stderr.decode(errors='replace')[-1000:]}")
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError(f"VIDEO_MISSING_OR_EMPTY:{self.path}")


def camera_rgb(camera: Any, width: int, height: int) -> np.ndarray:
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    rgb = rgb.astype(np.uint8, copy=False)
    if tuple(rgb.shape) != (height, width, 3):
        raise RuntimeError(f"CAMERA_RGB_SHAPE_MISMATCH:{tuple(rgb.shape)}")
    return rgb


class EvidenceVideo:
    def __init__(self, run_root: Path, env: Any) -> None:
        self.run_root = run_root
        self.front_camera = env.scene["safe_chest_front_camera"]
        self.side_camera = env.scene["safe_chest_side_camera"]
        self.width = int(self.front_camera.cfg.width)
        self.height = int(self.front_camera.cfg.height)
        self.front = VideoWriter(run_root / "safe_chest_joint_front.mp4", self.width, self.height)
        self.side = VideoWriter(run_root / "safe_chest_joint_side.mp4", self.width, self.height)
        self.font = ImageFont.load_default()
        self.counter = 0

    def _overlay(self, rgb: np.ndarray, record: dict[str, Any]) -> Image.Image:
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image, mode="RGBA")
        lines = (
            f"phase={record['phase']} frame={record['frame']}",
            f"target_source={record.get('target_source', 'S2-02_REFERENCE')}",
            f"limiting_joint={record.get('limiting_joint', 'NONE')}",
            f"min_margin={record['minimum_joint_limit_margin_rad']:.3f}",
            f"left_q_err={record['left_joint_tracking_max_error_rad']:.4f}",
            f"right_q_err={record['right_joint_tracking_max_error_rad']:.4f}",
            f"left_palm={record['left_palm_position_m']}",
            f"right_palm={record['right_palm_position_m']}",
            f"root_tilt={record['root_tilt_deg']:.3f} torque={record['arm_torque_ratio_max']:.3f}",
            f"collision={record['forbidden_collision']} hold={record.get('hold_timer_s', 0.0):.2f}",
        )
        height = 13
        draw.rectangle((3, 3, 440, 11 + height * len(lines)), fill=(0, 0, 0, 180))
        for index, line in enumerate(lines):
            draw.text((7, 6 + height * index), line, fill=(255, 255, 255, 255), font=self.font)
        return image

    def capture(self, record: dict[str, Any], *, force: bool = False, keyframe: str | None = None) -> None:
        front = self._overlay(camera_rgb(self.front_camera, self.width, self.height), record)
        side = self._overlay(camera_rgb(self.side_camera, self.width, self.height), record)
        if force or self.counter % 2 == 0:
            self.front.write(np.asarray(front, dtype=np.uint8))
            self.side.write(np.asarray(side, dtype=np.uint8))
        if keyframe is not None:
            front.save(self.run_root / f"frame_{keyframe}.png")
            side.save(self.run_root / f"frame_{keyframe}_side.png")
        self.counter += 1

    def close(self) -> dict[str, Any]:
        self.front.close()
        self.side.close()
        return {
            "front": {
                "path": str(self.front.path),
                "size_bytes": self.front.path.stat().st_size,
                "sha256": sha256(self.front.path),
                "encoded_frames": self.front.frames,
            },
            "side": {
                "path": str(self.side.path),
                "size_bytes": self.side.path.stat().st_size,
                "sha256": sha256(self.side.path),
                "encoded_frames": self.side.frames,
            },
        }


def load_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("TARGET_REFERENCE_MISSING")
    actual = sha256(path)
    if actual != REFERENCE_SHA256:
        raise RuntimeError(f"TARGET_REFERENCE_SHA_MISMATCH:{actual}")
    reference = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(reference, dict):
        raise RuntimeError("TARGET_REFERENCE_NOT_MAPPING")
    if list(reference.get("arm_joint_names", ())) != list(ARM_JOINT_NAMES):
        raise RuntimeError("TARGET_REFERENCE_ARM_ORDER_INVALID")
    if not isinstance(reference.get("arm_ik_target"), torch.Tensor):
        raise RuntimeError("TARGET_REFERENCE_ARM_Q_MISSING")
    if not finite(reference):
        raise RuntimeError("TARGET_REFERENCE_NONFINITE")
    source = reference.get("source", {})
    if source.get("controller_checkpoint_sha256") != LOWER_CHECKPOINT_SHA256:
        raise RuntimeError("TARGET_REFERENCE_LOWER_CHECKPOINT_SHA_MISMATCH")
    return reference


def recurrent_reset_audit(env: Any) -> dict[str, Any]:
    lower = env.action_manager.get_term("lower_body_joint_pos")
    policy = getattr(lower, "_policy", None)
    hidden = getattr(policy, "hidden_state", None)
    cell = getattr(policy, "cell_state", None)
    previous = getattr(lower, "_previous_policy_actions", None)
    if not all(isinstance(value, torch.Tensor) for value in (hidden, cell, previous)):
        return {"reset_verified": False, "reason": "RECURRENT_STATE_FIELDS_MISSING"}
    maxima = {
        "hidden_state_abs_max": float(hidden.detach().abs().max()),
        "cell_state_abs_max": float(cell.detach().abs().max()),
        "previous_policy_action_abs_max": float(previous.detach().abs().max()),
    }
    return {
        "reset_verified": all(value <= 1.0e-8 for value in maxima.values()),
        **maxima,
        "reason": "ZEROED" if all(value <= 1.0e-8 for value in maxima.values()) else "NONZERO_AFTER_RESET",
    }


def usd_hand_audit(stage: Any) -> dict[str, Any]:
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if not root.IsValid():
        raise RuntimeError("ROBOT_PRIM_MISSING")
    prims = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if any(token in path.lower() for token in ("hand", "finger", "palm")):
            prims.append({"path": path, "type": prim.GetTypeName()})
    return {
        "hand_visual_type": "Unitree G1 Dex3-1 hand (runtime USD prim audit)",
        "hardware_hand_match": "UNKNOWN",
        "hand_visual_prims": prims,
    }


def runtime_jacobian(robot: Any, body_id: int, joint_ids: list[int]) -> torch.Tensor:
    body_index = body_id if not robot.is_fixed_base else body_id - 1
    jacobian_joint_ids = [int(value) + 6 for value in joint_ids]
    raw = robot.root_physx_view.get_jacobians()[:, body_index, :, jacobian_joint_ids].clone()
    base_rotation = math_utils.matrix_from_quat(math_utils.quat_inv(robot.data.root_quat_w))
    raw[:, :3] = torch.bmm(base_rotation, raw[:, :3])
    raw[:, 3:] = torch.bmm(base_rotation, raw[:, 3:])
    return raw


def jacobian_metrics(robot: Any, body_ids: list[int], left_ids: list[int], right_ids: list[int]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for label, body_id, ids in (("left", body_ids[0], left_ids), ("right", body_ids[1], right_ids)):
        jac = runtime_jacobian(robot, body_id, ids)[0]
        singular = torch.linalg.svdvals(jac)
        condition = float(singular[0] / singular[-1].clamp_min(1.0e-12))
        values[label] = {
            "shape": list(jac.shape),
            "singular_values": tolist(singular),
            "condition_number": condition,
            "finite": bool(torch.isfinite(jac).all() and torch.isfinite(singular).all()),
        }
    values["dls_lambda"] = 0.01
    values["jacobian_frame"] = "pelvis/root frame; angular rows then linear rows from PhysX geometric Jacobian"
    return values


def runtime_audit(env: Any, cfg: Any, stage: Any) -> dict[str, Any]:
    robot = env.scene["robot"]
    arm_ids, arm_names = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
    left_ids, left_names = robot.find_joints(list(LEFT_ARM_JOINT_NAMES), preserve_order=True)
    right_ids, right_names = robot.find_joints(list(RIGHT_ARM_JOINT_NAMES), preserve_order=True)
    palm_ids, palm_names = robot.find_bodies(list(PALM_BODY_NAMES), preserve_order=True)
    wrist_ids, wrist_names = robot.find_bodies(
        ["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
    )
    frame_names = list(env.scene["hand_frames"].data.target_frame_names)
    if tuple(arm_names) != ARM_JOINT_NAMES:
        raise RuntimeError(f"JOINT_ORDER_INVALID:{arm_names}")
    if tuple(left_names) != LEFT_ARM_JOINT_NAMES or tuple(right_names) != RIGHT_ARM_JOINT_NAMES:
        raise RuntimeError("LEFT_RIGHT_JOINT_ORDER_INVALID")
    if tuple(palm_names) != PALM_BODY_NAMES or tuple(frame_names) != PALM_FRAME_NAMES:
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
        overlap[name] = [robot.joint_names[i] for i in sorted(set(ids) & set(int(v) for v in arm_ids))]
    expected_writers = {"left_arm_joint_pos", "right_arm_joint_pos"}
    overridden = any(bool(value) for name, value in overlap.items() if name not in expected_writers)

    lower = env.action_manager.get_term("lower_body_joint_pos")
    checkpoint_path = str(getattr(lower.cfg, "policy_path", ""))
    checkpoint_sha = sha256(Path(checkpoint_path)) if checkpoint_path and Path(checkpoint_path).is_file() else None
    if checkpoint_sha != LOWER_CHECKPOINT_SHA256:
        raise RuntimeError(f"LOWER_CHECKPOINT_SHA_MISMATCH:{checkpoint_sha}")
    limits = robot.data.joint_pos_limits[0, arm_ids].detach().clone()
    effort = robot.data.joint_effort_limits[0, arm_ids].detach().clone()
    body_paths = [
        "/World/envs/env_0/Robot/left_hand/left_hand_palm_link",
        "/World/envs/env_0/Robot/right_hand/right_hand_palm_link",
    ]
    return {
        "asset": {
            "usd_path": str(getattr(cfg.scene.robot.spawn, "usd_path", "METRIC_MISSING")),
            "urdf_path": "METRIC_MISSING",
            "robot_prim_path": "/World/envs/env_0/Robot",
        },
        "hand": usd_hand_audit(stage),
        "wrist_body_names": list(wrist_names),
        "wrist_body_ids": [int(v) for v in wrist_ids],
        "end_effector_body_names": list(palm_names),
        "end_effector_body_ids": [int(v) for v in palm_ids],
        "end_effector_body_paths": body_paths,
        "palm_frame_names": frame_names,
        "palm_tcp_contract": {
            "tcp_body": "palm_link (no separate TCP prim)",
            "palm_collision_support_offset_m": 0.44023889869451527,
            "local_normal_axis": list(S2_02_PALM_LOCAL_NORMAL_AXIS),
        },
        "arm_joint_names": list(arm_names),
        "arm_joint_ids": [int(v) for v in arm_ids],
        "left_arm_joint_names": list(left_names),
        "left_arm_joint_ids": [int(v) for v in left_ids],
        "right_arm_joint_names": list(right_names),
        "right_arm_joint_ids": [int(v) for v in right_ids],
        "arm_joint_limits_rad": tolist(limits),
        "arm_effort_limits": tolist(effort),
        "action_terms": {
            "names": names,
            "dimensions": dims,
            "slices": slices,
            "arm_overlap_by_term": overlap,
            "lower_body_joint_names": list(getattr(lower, "_joint_names", ())),
            "lower_body_checkpoint_path": checkpoint_path,
            "lower_body_checkpoint_sha256": checkpoint_sha,
            "lower_body_command": list(LOWER_BODY_COMMAND),
            "upper_body_target_source": ["left_arm_joint_pos", "right_arm_joint_pos"],
            "final_actuator_target_api": "JointPositionAction -> robot.set_joint_position_target",
        },
        "no_box_scene": True,
        "no_actor": True,
        "no_ppo": True,
        "no_planner": True,
        "no_base_command": True,
        "upper_body_target_overridden": overridden,
    }


def pose_to_homogeneous(position: torch.Tensor, quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Independent homogeneous-transform constructor (wxyz quaternion)."""

    w, x, y, z = quaternion_wxyz.unbind(-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion_wxyz.shape[:-1], 3, 3)
    result = torch.eye(4, dtype=position.dtype, device=position.device).expand(
        *position.shape[:-1], 4, 4
    ).clone()
    result[..., :3, :3] = rotation
    result[..., :3, 3] = position
    return result


def homogeneous_to_pose(transform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    position = transform[..., :3, 3]
    quaternion = math_utils.quat_from_matrix(transform[..., :3, :3])
    return position, quaternion


def explicit_target_pose(
    reference: dict[str, Any],
    current_root_pos_w: torch.Tensor,
    current_root_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute the frozen target with an explicit matrix chain.

    The production path uses IsaacLab ``combine_frame_transforms`` and
    ``subtract_frame_transforms``.  This path intentionally performs the same
    calculation with homogeneous matrices so a frame/order regression cannot
    hide inside one helper implementation.
    """

    device = current_root_pos_w.device
    dtype = current_root_pos_w.dtype
    ref_root = reference["robot_root_state_relative"].to(device=device, dtype=dtype)
    ref_box = reference["box_root_state_relative"].to(device=device, dtype=dtype)
    t_ref_root = pose_to_homogeneous(ref_root[:3], ref_root[3:7])
    t_ref_box = pose_to_homogeneous(ref_box[:3], ref_box[3:7])
    t_root_box = torch.linalg.inv(t_ref_root) @ t_ref_box
    t_world_root = pose_to_homogeneous(current_root_pos_w, current_root_quat_w)
    t_world_box = t_world_root @ t_root_box
    local = torch.tensor(certified_object_local_targets(), dtype=dtype, device=device)
    object_quat = torch.tensor(
        S2_02_PALM_QUATERNION_OBJECT_WXYZ,
        dtype=dtype,
        device=device,
    ).expand(2, 4)
    t_box_palm = pose_to_homogeneous(local, object_quat)
    t_world_palm = t_world_box.unsqueeze(0) @ t_box_palm
    t_root_palm = torch.linalg.inv(t_world_root).unsqueeze(0) @ t_world_palm
    position, quaternion = homogeneous_to_pose(t_root_palm)
    return position, quaternion, {
        "reference_box_relative_to_reference_root": tolist(t_root_box),
        "current_root_world": tolist(t_world_root),
        "current_box_world": tolist(t_world_box),
        "object_local_targets_m": certified_object_local_targets(),
        "object_quaternion_order": "wxyz",
        "target_frame": "ROOT",
    }


def production_target_pose(
    reference: dict[str, Any],
    current_root_pos_w: torch.Tensor,
    current_root_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    local = torch.tensor(
        certified_object_local_targets(), dtype=current_root_pos_w.dtype, device=current_root_pos_w.device
    )
    reference_root = reference["robot_root_state_relative"].to(
        device=current_root_pos_w.device, dtype=current_root_pos_w.dtype
    )
    reference_box = reference["box_root_state_relative"].to(
        device=current_root_pos_w.device, dtype=current_root_pos_w.dtype
    )
    box_pos_b, box_quat_b = math_utils.subtract_frame_transforms(
        reference_root[:3].unsqueeze(0), reference_root[3:7].unsqueeze(0),
        reference_box[:3].unsqueeze(0), reference_box[3:7].unsqueeze(0),
    )
    box_pos_w, box_quat_w = math_utils.combine_frame_transforms(
        current_root_pos_w.unsqueeze(0), current_root_quat_w.unsqueeze(0), box_pos_b, box_quat_b
    )
    object_quat = torch.tensor(
        S2_02_PALM_QUATERNION_OBJECT_WXYZ,
        dtype=current_root_pos_w.dtype,
        device=current_root_pos_w.device,
    ).expand(2, 4)
    world_pos, world_quat = math_utils.combine_frame_transforms(
        box_pos_w.expand(2, 3), box_quat_w.expand(2, 4), local, object_quat
    )
    target_pos, target_quat = math_utils.subtract_frame_transforms(
        current_root_pos_w.expand(2, 3), current_root_quat_w.expand(2, 4), world_pos, world_quat
    )
    return target_pos, target_quat, {
        "reference_box_relative_to_reference_root": {
            "position_m": tolist(box_pos_b[0]),
            "quaternion_wxyz": tolist(box_quat_b[0]),
        },
        "current_root_world": {
            "position_m": tolist(current_root_pos_w),
            "quaternion_wxyz": tolist(current_root_quat_w),
        },
        "current_box_world": {
            "position_m": tolist(box_pos_w[0]),
            "quaternion_wxyz": tolist(box_quat_w[0]),
        },
        "object_local_targets_m": certified_object_local_targets(),
        "object_quaternion_order": "wxyz",
        "target_frame": "ROOT",
    }


def target_frame_audit(
    reference: dict[str, Any],
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    production_pos, production_quat, production_meta = production_target_pose(reference, root_pos, root_quat)
    explicit_pos, explicit_quat, explicit_meta = explicit_target_pose(reference, root_pos, root_quat)
    position_diff = float((production_pos - explicit_pos).abs().max())
    rotation_diff = float(
        (math_utils.matrix_from_quat(production_quat) - math_utils.matrix_from_quat(explicit_quat)).abs().max()
    )
    audit = {
        "status": "PASS" if position_diff <= 1.0e-5 and rotation_diff <= 1.0e-5 else "FAIL",
        "production_path": S2_02_PELVIS_SYMBOL,
        "explicit_path": "homogeneous T_root_world^-1 @ T_box_world @ T_object_palm",
        "production_target_position_root_m": tolist(production_pos),
        "production_target_quaternion_root_wxyz": tolist(production_quat),
        "explicit_target_position_root_m": tolist(explicit_pos),
        "explicit_target_quaternion_root_wxyz": tolist(explicit_quat),
        "max_position_abs_diff_m": position_diff,
        "max_rotation_matrix_abs_diff": rotation_diff,
        "target_frame": "ROOT",
        "quaternion_order": "wxyz",
        "body_offset": None,
        "palm_collision_support_offset_m": 0.44023889869451527,
        "production_metadata": production_meta,
        "explicit_metadata": explicit_meta,
    }
    return production_pos, production_quat, audit


def palm_poses_in_root(env: Any, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    return math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        robot.data.body_pos_w[:, body_ids],
        robot.data.body_quat_w[:, body_ids],
    )


def set_arm_q(env: Any, arm_ids: list[int], q: torch.Tensor) -> None:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    joint_pos[:, arm_ids] = q.reshape(1, -1)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env.sim.forward()
    try:
        env.scene.update(0.0)
    except TypeError:
        env.scene.update(env.sim.dt)


def static_candidate_audit(
    env: Any,
    runtime: dict[str, Any],
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    candidates: dict[str, torch.Tensor],
) -> dict[str, Any]:
    robot = env.scene["robot"]
    arm_ids = [int(v) for v in runtime["arm_joint_ids"]]
    body_ids = [int(v) for v in runtime["end_effector_body_ids"]]
    limits = robot.data.joint_pos_limits[0, arm_ids]
    results: dict[str, Any] = {}
    for label, candidate in candidates.items():
        env.reset(seed=42)
        set_arm_q(env, arm_ids, candidate.to(device=env.device))
        actual_pos, actual_quat = palm_poses_in_root(env, body_ids)
        pos_err = actual_pos[0] - target_pos
        ori_err = math_utils.compute_pose_error(actual_pos, actual_quat, target_pos, target_quat, rot_error_type="axis_angle")[1]
        margins = torch.minimum(candidate - limits[:, 0], limits[:, 1] - candidate)
        collision, force_max, bodies = forbidden_contact(env.scene["contact_forces"])
        jac = jacobian_metrics(robot, body_ids, [int(v) for v in runtime["left_arm_joint_ids"]], [int(v) for v in runtime["right_arm_joint_ids"]])
        symmetry = float(abs(float(actual_pos[0, 1] + actual_pos[1, 1])))
        limiting_index = int(torch.argmin(margins))
        results[label] = {
            "candidate_q_rad": tolist(candidate),
            "position_residual_left_m": float(torch.linalg.vector_norm(pos_err[0])),
            "position_residual_right_m": float(torch.linalg.vector_norm(pos_err[1])),
            "orientation_residual_left_deg": math.degrees(float(torch.linalg.vector_norm(ori_err[0]))),
            "orientation_residual_right_deg": math.degrees(float(torch.linalg.vector_norm(ori_err[1]))),
            "fk_position_root_m": tolist(actual_pos),
            "fk_quaternion_root_wxyz": tolist(actual_quat),
            "minimum_joint_margin_rad": float(margins.min()),
            "limiting_joint": ARM_JOINT_NAMES[limiting_index],
            "collision_free": collision is False,
            "forbidden_collision": collision,
            "forbidden_contact_force_max_n": force_max,
            "forbidden_contact_bodies": bodies,
            "symmetry_y_residual_m": symmetry,
            "jacobian": jac,
            "finite": finite(candidate) and finite(actual_pos) and finite(actual_quat),
        }
        results[label]["safe"] = bool(
            results[label]["finite"]
            and results[label]["minimum_joint_margin_rad"] >= STATIC_MARGIN_GATE_RAD
            and results[label]["collision_free"]
            and max(results[label]["position_residual_left_m"], results[label]["position_residual_right_m"]) <= 0.10
            and symmetry <= 0.10
        )
    return results


def set_camera_views(env: Any) -> None:
    # Three-quarter front: chest, both arms, pelvis and feet remain in frame.
    env.scene["safe_chest_front_camera"].set_world_poses_from_view(
        torch.tensor([[2.20, -2.80, 1.35]], device=env.device),
        torch.tensor([[0.0, 0.0, 0.84]], device=env.device),
    )
    # Side view: the arm sweep and palm-to-body clearance are visible.
    env.scene["safe_chest_side_camera"].set_world_poses_from_view(
        torch.tensor([[2.80, 0.05, 1.12]], device=env.device),
        torch.tensor([[0.0, 0.0, 0.86]], device=env.device),
    )


def action_layout(env: Any) -> dict[str, slice]:
    names = list(env.action_manager.active_terms)
    dims = [int(value) for value in env.action_manager.action_term_dim]
    result: dict[str, slice] = {}
    cursor = 0
    for name, dim in zip(names, dims, strict=True):
        result[name] = slice(cursor, cursor + dim)
        cursor += dim
    required = {
        "left_arm_joint_pos",
        "right_arm_joint_pos",
        "waist_joint_pos",
        "finger_joint_pos",
        "lower_body_joint_pos",
    }
    if not required.issubset(result):
        raise RuntimeError(f"ACTION_LAYOUT_INVALID:{names}")
    result["__total__"] = slice(0, cursor)
    return result


def make_record(
    env: Any,
    runtime: dict[str, Any],
    phase: str,
    frame: int,
    desired_q: torch.Tensor,
    target_q: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    baseline_root: torch.Tensor,
    previous_target: torch.Tensor | None,
    hold_timer_s: float,
    command_clip_applied: bool,
    recurrent_reset: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    robot = env.scene["robot"]
    arm_ids = [int(v) for v in runtime["arm_joint_ids"]]
    body_ids = [int(v) for v in runtime["end_effector_body_ids"]]
    actual_q = robot.data.joint_pos[0, arm_ids].detach().clone()
    actuator_target = robot.data.joint_pos_target[0, arm_ids].detach().clone()
    velocity = robot.data.joint_vel[0, arm_ids].detach().clone()
    limits = robot.data.joint_pos_limits[0, arm_ids]
    margins = torch.minimum(actual_q - limits[:, 0], limits[:, 1] - actual_q)
    target_margins = torch.minimum(actuator_target - limits[:, 0], limits[:, 1] - actuator_target)
    effort = robot.data.joint_effort_limits[0, arm_ids].abs().clamp_min(1.0e-6)
    torque_ratio = (robot.data.applied_torque[0, arm_ids].abs() / effort).max()
    actual_pos, actual_quat = palm_poses_in_root(env, body_ids)
    pose_error = math_utils.compute_pose_error(
        actual_pos, actual_quat, target_pos, target_quat, rot_error_type="axis_angle"
    )
    collision, force_max, bodies = forbidden_contact(env.scene["contact_forces"])
    root_pos = robot.data.root_link_pos_w[0]
    root_quat = robot.data.root_link_quat_w[0]
    roll, pitch, yaw, tilt = root_metrics(root_quat)
    q_error = (actual_q - desired_q).abs()
    target_rate = 0.0 if previous_target is None else float((actuator_target - previous_target).abs().max() / CONTROL_DT_S)
    limiting_index = int(torch.argmin(margins))
    finite_state = bool(
        torch.isfinite(actual_q).all()
        and torch.isfinite(actuator_target).all()
        and torch.isfinite(velocity).all()
        and torch.isfinite(actual_pos).all()
        and torch.isfinite(actual_quat).all()
        and torch.isfinite(root_pos).all()
        and torch.isfinite(root_quat).all()
        and torch.isfinite(robot.data.applied_torque[0, arm_ids]).all()
        and collision is not None
    )
    record = {
        "frame": int(frame),
        "time_s": float(frame * CONTROL_DT_S),
        "phase": phase,
        "target_source": "S2-02 precontact_reference.pt arm_ik_target",
        "finite": finite_state,
        "desired_upper_body_joints_rad": tolist(desired_q),
        "target_upper_body_joints_rad": tolist(target_q),
        "actual_upper_body_joints_rad": tolist(actual_q),
        "final_actuator_target_rad": tolist(actuator_target),
        "upper_joint_velocity_rad_s": tolist(velocity),
        "upper_joint_velocity_max_rad_s": float(velocity.abs().max()),
        "left_joint_tracking_max_error_rad": float(q_error[0::2].max()),
        "right_joint_tracking_max_error_rad": float(q_error[1::2].max()),
        "action_rate_max_rad_s": target_rate,
        "action_rate_source": "absolute JointPositionAction target delta / control_dt",
        "command_clip_applied": bool(command_clip_applied),
        "arm_torque_ratio_max": float(torque_ratio),
        "minimum_joint_limit_margin_rad": float(margins.min()),
        "minimum_target_joint_limit_margin_rad": float(target_margins.min()),
        "limiting_joint": ARM_JOINT_NAMES[limiting_index],
        "left_palm_position_m": tolist(actual_pos[0]),
        "right_palm_position_m": tolist(actual_pos[1]),
        "left_palm_quaternion_wxyz": tolist(actual_quat[0]),
        "right_palm_quaternion_wxyz": tolist(actual_quat[1]),
        "left_position_error_m": float(torch.linalg.vector_norm(pose_error[0][0])),
        "right_position_error_m": float(torch.linalg.vector_norm(pose_error[0][1])),
        "left_position_error_xyz_m": tolist(pose_error[0][0]),
        "right_position_error_xyz_m": tolist(pose_error[0][1]),
        "left_orientation_error_deg": math.degrees(float(torch.linalg.vector_norm(pose_error[1][0]))),
        "right_orientation_error_deg": math.degrees(float(torch.linalg.vector_norm(pose_error[1][1]))),
        "root_position_world_m": tolist(root_pos),
        "root_displacement_xy_m": float(torch.linalg.vector_norm(root_pos[:2] - baseline_root[:2])),
        "root_displacement_xyz_m": float(torch.linalg.vector_norm(root_pos - baseline_root)),
        "root_height_m": float(root_pos[2]),
        "root_roll_deg": math.degrees(roll),
        "root_pitch_deg": math.degrees(pitch),
        "root_yaw_deg": math.degrees(yaw),
        "root_tilt_deg": math.degrees(tilt),
        "forbidden_collision": bool(collision) if collision is not None else None,
        "forbidden_contact_force_max_n": force_max,
        "forbidden_contact_bodies": bodies,
        "hold_timer_s": float(hold_timer_s),
        "recurrent_reset": recurrent_reset,
        "upper_body_target_overridden": bool(runtime["upper_body_target_overridden"]),
        "joint_order_swapped": False,
        "ik_frame_error": False,
        "pose_delta_accumulation_detected": False,
        "action_source": {
            "left_arm": "action_manager.left_arm_joint_pos:JointPositionAction",
            "right_arm": "action_manager.right_arm_joint_pos:JointPositionAction",
            "lower_body": "action_manager.lower_body_joint_pos:frozen_recurrent_student",
            "base": "fixed_zero_command",
        },
    }
    return record, actuator_target


def detect_oscillation(records: list[dict[str, Any]], phase: str) -> bool:
    selected = [item for item in records if item["phase"] == phase]
    if len(selected) < 12:
        return False
    points = np.asarray(
        [[*item["left_palm_position_m"], *item["right_palm_position_m"]] for item in selected],
        dtype=float,
    )
    velocity = np.diff(points, axis=0) / CONTROL_DT_S
    for axis in range(points.shape[1]):
        values = velocity[:, axis]
        values = values[np.abs(values) > 1.0e-4]
        reversals = int(np.sum(np.sign(values[1:]) * np.sign(values[:-1]) < 0)) if len(values) > 1 else 0
        amplitude = float(points[:, axis].max() - points[:, axis].min())
        if reversals >= 8 and amplitude > 0.003:
            return True
    return False


def run_episode(
    env: Any,
    runtime: dict[str, Any],
    *,
    label: str,
    target_q: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    hold_steps: int,
    active: tuple[bool, bool],
    formal: bool,
    video: EvidenceVideo | None,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=42)
    if not finite(observation):
        raise RuntimeError(f"OBSERVATION_NONFINITE_AFTER_RESET:{label}")
    reset_audit = recurrent_reset_audit(env)
    if not reset_audit["reset_verified"]:
        raise RuntimeError(f"RECURRENT_RESET_INVALID:{label}:{reset_audit}")
    robot = env.scene["robot"]
    arm_ids = [int(v) for v in runtime["arm_joint_ids"]]
    waist_ids, _ = robot.find_joints(["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"], preserve_order=True)
    finger_names = [name for name in robot.joint_names if "hand_" in name]
    finger_ids, _ = robot.find_joints(finger_names, preserve_order=True)
    layout = action_layout(env)
    q0 = robot.data.joint_pos[0, arm_ids].detach().clone()
    waist_target = robot.data.default_joint_pos[0, waist_ids].detach().clone()
    finger_target = robot.data.default_joint_pos[0, finger_ids].detach().clone()
    baseline_root = robot.data.root_link_pos_w[0].detach().clone()
    desired_target = q0.clone()
    if active[0]:
        desired_target[0::2] = target_q[0::2]
    if active[1]:
        desired_target[1::2] = target_q[1::2]
    actions = torch.zeros((1, int(layout["__total__"].stop)), dtype=torch.float32, device=env.device)
    lower_command = torch.tensor(LOWER_BODY_COMMAND, dtype=torch.float32, device=env.device)
    records: list[dict[str, Any]] = []
    previous_target: torch.Tensor | None = None
    previous_command = q0.clone()
    early_done = False
    safety_trigger: str | None = None
    clip_count = 0

    if video is not None:
        initial, initial_target = make_record(
            env, runtime, "INITIAL", 0, q0, target_q, target_pos, target_quat,
            baseline_root, None, 0.0, False, reset_audit,
        )
        initial["desired_upper_body_joints_rad"] = tolist(initial_target)
        video.capture(initial, force=True, keyframe="initial" if label == "DEFAULT_ARMS_STAND_BASELINE" else None)

    def one_step(phase: str, fraction: float, frame: int, hold_timer: float) -> None:
        nonlocal previous_target, previous_command, early_done, safety_trigger, clip_count
        raw_command = q0 + float(minimum_jerk_fraction(int(fraction * MOVE_STEPS), MOVE_STEPS)) * (desired_target - q0)
        if phase.endswith("HOLD") or phase == "BASELINE":
            raw_command = desired_target
        max_delta = JOINT_RATE_LIMIT_RAD_S * CONTROL_DT_S
        delta = raw_command - previous_command
        command = previous_command + torch.clamp(delta, -max_delta, max_delta)
        clipped = bool((command - raw_command).abs().max() > 1.0e-8)
        if clipped:
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
        record, previous_target = make_record(
            env, runtime, phase, frame, command, target_q, target_pos, target_quat,
            baseline_root, previous_target, hold_timer, clipped, reset_audit,
        )
        records.append(record)
        previous_command = command.detach().clone()
        trigger = None
        if not record["finite"]:
            trigger = "METRIC_OR_STATE_NONFINITE"
        elif record["forbidden_collision"] is True:
            trigger = "FORBIDDEN_COLLISION"
        elif not ROOT_HEIGHT_RANGE_M[0] <= record["root_height_m"] <= ROOT_HEIGHT_RANGE_M[1]:
            trigger = "ROOT_HEIGHT"
        elif record["root_tilt_deg"] > ROOT_TILT_GATE_DEG:
            trigger = "ROOT_TILT"
        elif record["minimum_joint_limit_margin_rad"] < JOINT_MARGIN_GATE_RAD:
            trigger = "ACTUAL_JOINT_MARGIN"
        elif record["minimum_target_joint_limit_margin_rad"] < JOINT_MARGIN_GATE_RAD:
            trigger = "TARGET_JOINT_MARGIN"
        elif record["arm_torque_ratio_max"] > 1.001:
            trigger = "ARM_TORQUE_RATIO"
        if trigger is not None:
            safety_trigger = trigger
            early_done = True
        if video is not None:
            video.capture(record, force=False)
        if frame % 25 == 0 or frame == 1:
            print(f"PHASE={phase} step={frame}", flush=True)

    for frame in range(1, SETTLE_STEPS + 1):
        one_step("BASELINE" if label == "DEFAULT_ARMS_STAND_BASELINE" else "SETTLE", 0.0, frame, 0.0)
        if early_done:
            break
    if not early_done and label != "DEFAULT_ARMS_STAND_BASELINE":
        for index in range(1, MOVE_STEPS + 1):
            frame = SETTLE_STEPS + index
            one_step("MOVE", index / MOVE_STEPS, frame, 0.0)
            if early_done:
                break
    if not early_done:
        for index in range(1, hold_steps + 1):
            frame = SETTLE_STEPS + (0 if label == "DEFAULT_ARMS_STAND_BASELINE" else MOVE_STEPS) + index
            one_step("HOLD", 1.0, frame, index * CONTROL_DT_S)
            if video is not None and index == 1:
                video.capture(records[-1], force=True, keyframe="precontact")
            if early_done:
                break

    oscillation = detect_oscillation(records, "HOLD")
    for item in records:
        item["oscillation_detected"] = oscillation
    if video is not None and records:
        video.capture(records[-1], force=True, keyframe="terminal")
    hold_records = [item for item in records if item["phase"] == "HOLD"]
    safety_ok = bool(records) and all(
        item["finite"]
        and item["forbidden_collision"] is False
        and ROOT_HEIGHT_RANGE_M[0] <= item["root_height_m"] <= ROOT_HEIGHT_RANGE_M[1]
        and item["root_tilt_deg"] <= ROOT_TILT_GATE_DEG
        and item["minimum_joint_limit_margin_rad"] >= JOINT_MARGIN_GATE_RAD
        and item["minimum_target_joint_limit_margin_rad"] >= JOINT_MARGIN_GATE_RAD
        and item["arm_torque_ratio_max"] <= 1.001
        for item in records
    )
    return {
        "label": label,
        "active_arms": {"left": active[0], "right": active[1]},
        "observed_control_frames": len(records),
        "expected_control_frames": SETTLE_STEPS + (0 if label == "DEFAULT_ARMS_STAND_BASELINE" else MOVE_STEPS) + hold_steps,
        "hold_frames": len(hold_records),
        "hold_seconds": len(hold_records) * CONTROL_DT_S,
        "early_done": early_done,
        "safety_trigger": safety_trigger,
        "safety_ok": safety_ok,
        "recurrent_reset": reset_audit,
        "oscillation_detected": oscillation,
        "command_clip_count": clip_count,
        "left_position_error_p95_m": percentile([item["left_position_error_m"] for item in records], 95),
        "right_position_error_p95_m": percentile([item["right_position_error_m"] for item in records], 95),
        "left_position_error_max_m": max((item["left_position_error_m"] for item in records), default=None),
        "right_position_error_max_m": max((item["right_position_error_m"] for item in records), default=None),
        "root_tilt_max_deg": max((item["root_tilt_deg"] for item in records), default=None),
        "root_height_min_m": min((item["root_height_m"] for item in records), default=None),
        "root_height_max_m": max((item["root_height_m"] for item in records), default=None),
        "root_displacement_xy_max_m": max((item["root_displacement_xy_m"] for item in records), default=None),
        "max_torque_ratio": max((item["arm_torque_ratio_max"] for item in records), default=None),
        "tracking_ok": bool(hold_records) and not oscillation,
        "records": records,
    }


def write_trace(label: str, result: dict[str, Any]) -> None:
    write_json(RUN / f"{label.lower()}_trace.json", {"stage": STAGE, "label": label, "records": result["records"]})


def old_runtime_jacobian_audit(
    env: Any,
    runtime: dict[str, Any],
    old_audit: dict[str, Any],
) -> dict[str, Any]:
    """Instrument the same PhysX Jacobian API at the old terminal arm q.

    The old trace did not persist these values.  We therefore label the
    measurement explicitly as a runtime re-evaluation at the recorded q,
    rather than pretending it was part of the historical trace.
    """

    robot = env.scene["robot"]
    arm_ids = [int(v) for v in runtime["arm_joint_ids"]]
    left_ids = [int(v) for v in runtime["left_arm_joint_ids"]]
    right_ids = [int(v) for v in runtime["right_arm_joint_ids"]]
    body_ids = [int(v) for v in runtime["end_effector_body_ids"]]
    saved_q = robot.data.joint_pos.clone()
    saved_v = robot.data.joint_vel.clone()
    try:
        old_q = torch.tensor(
            old_audit["target_margin_history"][-1]["target_margin_by_joint_rad"],
            dtype=robot.data.joint_pos.dtype,
            device=env.device,
        )
        # The history is a mapping; use the exact recorded requested vector
        # from the trace instead of reconstructing it from margins.
        trace = json.loads(OLD_TRACE_PATH.read_text(encoding="utf-8"))
        old_q = torch.tensor(
            trace["records"][-1]["desired_upper_body_joints_rad"],
            dtype=robot.data.joint_pos.dtype,
            device=env.device,
        )
        set_arm_q(env, arm_ids, old_q)
        metrics = jacobian_metrics(robot, body_ids, left_ids, right_ids)
        metrics["evaluation_q_source"] = "old formal_trace.json frame=150 desired_upper_body_joints_rad"
        metrics["trace_persisted_singular_values"] = False
        metrics["trace_persisted_condition_number"] = False
        return metrics
    except Exception as exc:  # runtime API differences are evidence, not a guess
        return {
            "status": "METRIC_MISSING",
            "reason": f"RUNTIME_JACOBIAN_AUDIT_EXCEPTION:{type(exc).__name__}:{exc}",
            "dls_lambda": 0.01,
        }
    finally:
        try:
            robot.write_joint_state_to_sim(saved_q, saved_v)
            env.sim.forward()
            env.scene.update(0.0)
        except Exception:
            pass


def classify_result(
    baseline: dict[str, Any],
    static_selected: dict[str, Any] | None,
    formal: dict[str, Any] | None,
    target_audit: dict[str, Any],
    videos: dict[str, Any] | None,
) -> tuple[str, str]:
    if target_audit.get("status") != "PASS":
        return "INVALID", "CHEST_TARGET_FRAME_TRANSFORM_INVALID"
    if not baseline.get("safety_ok") or baseline.get("hold_seconds", 0.0) < 10.0:
        return "FAIL", "DEFAULT_ARMS_STAND_BASELINE_FAILED"
    if static_selected is None or not static_selected.get("safe"):
        return "FAIL", "SAFE_REFERENCE_KINEMATIC_GATE_FAILED"
    if formal is None:
        return "INVALID", "FORMAL_TRACE_MISSING"
    if formal.get("early_done") and formal.get("safety_trigger"):
        return "FAIL", "SAFETY_GATE_TRIGGERED"
    if formal.get("hold_seconds", 0.0) < FORMAL_HOLD_SECONDS:
        return "INVALID", "FORMAL_HOLD_INCOMPLETE"
    if not formal.get("safety_ok"):
        return "FAIL", "SAFETY_GATE_TRIGGERED"
    if formal.get("oscillation_detected"):
        return "FAIL", "CHEST_JOINT_TRACKING_UNSTABLE"
    if videos is None or any(int(item.get("size_bytes", 0)) <= 0 for item in videos.values()):
        return "INVALID", "VISUAL_EVIDENCE_INCOMPLETE"
    return "PASS", "SAFE_CHEST_JOINT_REFERENCE_AND_HOLD_COMPLETE"


def main() -> int:
    global ENV, VIDEO_RECORDER
    reference_path = (args.reference or REFERENCE_PATH).resolve()
    write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": "OLD_TRACE_AUDIT"})
    reference = load_reference(reference_path)
    old_trace = json.loads(OLD_TRACE_PATH.read_text(encoding="utf-8"))
    old_geometry = json.loads(OLD_GEOMETRY_AUDIT_PATH.read_text(encoding="utf-8"))
    old_audit = audit_old_trace(old_trace, old_geometry)
    write_json(RUN / "old_trace_audit.json", old_audit)

    cfg = S203TSafeChestJointReferenceEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    ENV = ManagerBasedEnv(cfg=cfg)
    write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": "ENVIRONMENT_CREATED"})
    set_camera_views(ENV)
    observation, _ = ENV.reset(seed=args.seed)
    if not finite(observation):
        raise RuntimeError("OBSERVATION_NONFINITE_AFTER_INITIAL_RESET")
    runtime = runtime_audit(ENV, cfg, get_current_stage())
    robot = ENV.scene["robot"]
    root_pos = robot.data.root_link_pos_w[0].detach().clone()
    root_quat = robot.data.root_link_quat_w[0].detach().clone()
    target_pos, target_quat, target_audit = target_frame_audit(reference, root_pos, root_quat)
    write_json(RUN / "target_frame_audit.json", target_audit)
    old_audit["runtime_jacobian"] = old_runtime_jacobian_audit(ENV, runtime, old_audit)
    write_json(RUN / "old_trace_audit.json", old_audit)

    arm_ids = [int(v) for v in runtime["arm_joint_ids"]]
    q_reference = reference["arm_ik_target"].to(device=ENV.device, dtype=robot.data.joint_pos.dtype).detach().clone()
    q_default = robot.data.default_joint_pos[0, arm_ids].detach().clone()
    # Ablations are evaluated as bounded static candidates.  The certified
    # S2-02 q is the only candidate that carries the full orientation target;
    # the other two deliberately relax it toward the verified default.
    candidates = {
        "POSITION_ONLY": q_default,
        "RELAXED_ORIENTATION": 0.75 * q_reference + 0.25 * q_default,
        "FULL_S2_02_ORIENTATION": q_reference,
    }
    write_json(
        RUN / "resolved_config.json",
        {
            "schema_version": 1,
            "stage": STAGE,
            "seed": args.seed,
            "num_envs": 1,
            "control_dt_s": CONTROL_DT_S,
            "move_seconds": MOVE_STEPS * CONTROL_DT_S,
            "hold_seconds": FORMAL_HOLD_SECONDS,
            "render_mode": "rgb_array",
            "enable_cameras": True,
            "box_used": False,
            "contact_test_started": False,
            "base_walking_started": False,
            "ppo_started": False,
            "control_mode": "JOINT_SPACE_MINIMUM_JERK",
            "differential_ik_used_for_formal_move": False,
            "target_source": "S2-02 precontact_reference.pt arm_ik_target",
            "reference_path": str(reference_path),
            "reference_sha256": sha256(reference_path),
            "lower_checkpoint_sha256": LOWER_CHECKPOINT_SHA256,
            "lower_body_command": list(LOWER_BODY_COMMAND),
            "target_frame_audit": target_audit,
            "runtime": runtime,
            "old_trace_audit_path": str(RUN / "old_trace_audit.json"),
            "candidates_q_rad": {label: tolist(q) for label, q in candidates.items()},
            "thresholds": {
                "static_min_margin_rad": STATIC_MARGIN_GATE_RAD,
                "dynamic_min_margin_rad": JOINT_MARGIN_GATE_RAD,
                "root_tilt_deg": ROOT_TILT_GATE_DEG,
                "torque_ratio": 1.001,
                "forbidden_force_n": FORBIDDEN_FORCE_GATE_N,
            },
        },
    )

    write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": "DEFAULT_ARMS_STAND_BASELINE"})
    VIDEO_RECORDER = EvidenceVideo(RUN, ENV)
    baseline = run_episode(
        ENV,
        runtime,
        label="DEFAULT_ARMS_STAND_BASELINE",
        target_q=q_default,
        target_pos=target_pos,
        target_quat=target_quat,
        hold_steps=500,
        active=(False, False),
        formal=False,
        video=VIDEO_RECORDER,
    )
    write_trace("DEFAULT_ARMS_STAND_BASELINE", baseline)
    if not baseline["safety_ok"] or baseline["hold_seconds"] < 10.0:
        videos = VIDEO_RECORDER.close()
        VIDEO_RECORDER = None
        status, reason = "FAIL", "DEFAULT_ARMS_STAND_BASELINE_FAILED"
        result = {
            "schema_version": 1,
            "stage": STAGE,
            "status": status,
            "primary_reason": reason,
            "old_trace_audit": old_audit,
            "target_frame_audit": target_audit,
            "runtime_audit": runtime,
            "baseline": {k: v for k, v in baseline.items() if k != "records"},
            "static_reference": None,
            "formal": None,
            "videos": videos,
            "scientific_contract": {"box_used": False, "contact_test_started": False, "base_walking_started": False, "ppo_started": False},
        }
        write_json(RUN / "result.json", result)
        write_json(RUN / "summary.json", {"stage": STAGE, "status": status, "primary_reason": reason, "baseline": result["baseline"], "videos": videos})
        print(f"CHEST_REFERENCE_RECOVERY_STATUS={status}", flush=True)
        print(f"PRIMARY_REASON={reason}", flush=True)
        return 0

    write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": "STATIC_CANDIDATE_AUDIT"})
    static_results = static_candidate_audit(ENV, runtime, target_pos, target_quat, candidates)
    write_json(RUN / "static_candidate_audit.json", static_results)
    safe_labels = [label for label, value in static_results.items() if value.get("safe")]
    selected_label = min(
        safe_labels,
        key=lambda label: max(static_results[label]["position_residual_left_m"], static_results[label]["position_residual_right_m"]),
        default=None,
    )
    selected = static_results.get(selected_label) if selected_label is not None else None
    selected_q = candidates[selected_label] if selected_label is not None else None
    write_json(
        RUN / "selected_joint_reference.json",
        {
            "source": "S2_02_OBSERVED_STABLE_ARM_Q",
            "provenance_note": "S2-02 candidate-9 reference replay; S2-02 trace did not persist 14D q vectors",
            "candidate_label": selected_label,
            "q_rad": tolist(selected_q) if selected_q is not None else None,
            "metrics": selected,
            "status": "KINEMATICALLY_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED" if selected is not None else "NONE",
        },
    )

    probes: list[dict[str, Any]] = []
    if selected_q is not None:
        for label, active in (("LEFT_ONLY", (True, False)), ("RIGHT_ONLY", (False, True)), ("BILATERAL", (True, True))):
            write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": label})
            probe = run_episode(
                ENV,
                runtime,
                label=label,
                target_q=selected_q,
                target_pos=target_pos,
                target_quat=target_quat,
                hold_steps=SHORT_HOLD_STEPS,
                active=active,
                formal=False,
                video=None,
            )
            write_trace(label, probe)
            probes.append(probe)

    formal: dict[str, Any] | None = None
    if selected_q is not None:
        write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": "FORMAL_JOINT_SPACE_MOVE_HOLD"})
        formal = run_episode(
            ENV,
            runtime,
            label="FORMAL_JOINT_SPACE_MOVE_HOLD",
            target_q=selected_q,
            target_pos=target_pos,
            target_quat=target_quat,
            hold_steps=FORMAL_HOLD_STEPS,
            active=(True, True),
            formal=True,
            video=VIDEO_RECORDER,
        )
        write_trace("FORMAL_JOINT_SPACE_MOVE_HOLD", formal)
    videos = VIDEO_RECORDER.close()
    VIDEO_RECORDER = None
    status, reason = classify_result(baseline, selected, formal, target_audit, videos)
    result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": status,
        "primary_reason": reason,
        "old_target_can_pass_0p10_margin_gate": old_audit["old_target_can_pass_0p10_margin_gate"],
        "old_trace_audit": old_audit,
        "target_frame_audit": target_audit,
        "runtime_audit": runtime,
        "baseline": {k: v for k, v in baseline.items() if k != "records"},
        "static_candidates": static_results,
        "selected_joint_reference": {
            "source": "S2_02_OBSERVED_STABLE_ARM_Q",
            "candidate_label": selected_label,
            "q_rad": tolist(selected_q) if selected_q is not None else None,
            "status": "KINEMATICALLY_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED" if selected_q is not None else "NONE",
        },
        "probes": [{k: v for k, v in item.items() if k != "records"} for item in probes],
        "formal": {k: v for k, v in formal.items() if k != "records"} if formal is not None else None,
        "videos": videos,
        "trajectory": {
            "control_mode": "JOINT_SPACE_MINIMUM_JERK",
            "settle_steps": SETTLE_STEPS,
            "move_steps": MOVE_STEPS,
            "hold_steps": FORMAL_HOLD_STEPS,
            "hold_seconds": FORMAL_HOLD_SECONDS,
            "differential_ik_used_for_formal_move": False,
            "rate_limit_rad_s": JOINT_RATE_LIMIT_RAD_S,
        },
        "scientific_contract": {
            "box_used": False,
            "contact_test_started": False,
            "base_walking_started": False,
            "ppo_started": False,
            "falcon_started": False,
            "planner_started": False,
        },
        "authoritative_complete_before_teardown": True,
    }
    write_json(RUN / "result.json", result)
    write_json(
        RUN / "summary.json",
        {
            "stage": STAGE,
            "status": status,
            "primary_reason": reason,
            "selected_candidate": selected_label,
            "old_limiting_joint": old_audit["limiting_joint"],
            "old_target_margin_rad": old_audit["target_margin_rad"],
            "old_actual_margin_rad": old_audit["actual_margin_rad"],
            "baseline_hold_seconds": baseline["hold_seconds"],
            "formal_hold_seconds": formal["hold_seconds"] if formal else None,
            "formal_root_tilt_max_deg": formal["root_tilt_max_deg"] if formal else None,
            "formal_safety_ok": formal["safety_ok"] if formal else False,
            "videos": videos,
        },
    )
    write_json(RUN / "runner_status.json", {"status": "COMPLETE", "phase": "RESULT_WRITTEN", "primary_reason": reason})
    print(f"CHEST_REFERENCE_RECOVERY_STATUS={status}", flush=True)
    print(f"PRIMARY_REASON={reason}", flush=True)
    print(f"OLD_TARGET_CAN_PASS_0P10_MARGIN_GATE={old_audit['old_target_can_pass_0p10_margin_gate']}", flush=True)
    print(f"OLD_LIMITING_JOINT={old_audit['limiting_joint']}", flush=True)
    print(f"OLD_TARGET_MARGIN_RAD={old_audit['target_margin_rad']}", flush=True)
    print(f"OLD_ACTUAL_MARGIN_RAD={old_audit['actual_margin_rad']}", flush=True)
    print(f"DEFAULT_ARMS_STAND_BASELINE={'PASS' if baseline['safety_ok'] and baseline['hold_seconds'] >= 10.0 else 'FAIL'}", flush=True)
    print(f"SAFE_REFERENCE_SOURCE={'S2_02_OBSERVED_STABLE_ARM_Q' if selected_q is not None else 'NONE'}", flush=True)
    print("CHEST_PREPOSE_CONTROL_MODE=JOINT_SPACE_MINIMUM_JERK", flush=True)
    print("DIFFERENTIAL_IK_USED_FOR_FORMAL_MOVE=NO", flush=True)
    print(f"HOLD_SECONDS={formal['hold_seconds'] if formal else 0.0}", flush=True)
    return 0


try:
    rc = main()
except BaseException as exc:
    payload = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "INVALID",
        "primary_reason": "IMPLEMENTATION_EXCEPTION",
        "authoritative_complete_before_teardown": False,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    write_json(RUN / "implementation_exception.json", payload)
    # Do not overwrite a complete authoritative result with a teardown error.
    if not (RUN / "result.json").is_file():
        write_json(RUN / "result.json", payload)
    write_json(RUN / "runner_status.json", {"status": "INVALID", "phase": "EXCEPTION", "primary_reason": "IMPLEMENTATION_EXCEPTION"})
    print("CHEST_REFERENCE_RECOVERY_STATUS=INVALID", flush=True)
    print("PRIMARY_REASON=IMPLEMENTATION_EXCEPTION", flush=True)
    rc = 1
finally:
    if VIDEO_RECORDER is not None:
        try:
            VIDEO_RECORDER.close()
        except Exception:
            pass
    try:
        simulation_app.close()
    except Exception:
        pass
