#!/usr/bin/env python3
"""Deterministic no-box S2-03T chest-prepose qualification."""

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
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


write_json(RUN / "runner_status.json", {"status": "STARTING", "phase": "APP_LAUNCH"})

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--reference", type=Path)
parser.add_argument("--seed", type=int, default=42)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seed != 42:
    raise SystemExit("CHEST_PREPOSE_SEED_MUST_BE_42")
simulation_app = AppLauncher(args).app

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Usd  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

from g1_access_push.sim.stage2.s2_03t_chest_prepose_env_cfg import (  # noqa: E402
    S203TChestPreposeEnvCfg,
)
from g1_access_push.stage2.s2_03t_chest_prepose_contract import (  # noqa: E402
    ARM_JOINT_NAMES,
    CONTROL_DT_S,
    FORBIDDEN_FORCE_GATE_N,
    HOLD_SECONDS,
    HOLD_STEPS,
    LEFT_ARM_JOINT_NAMES,
    MAX_ORIENTATION_COMMAND_RAD,
    MAX_POSITION_COMMAND_M,
    MOVE_STEPS,
    PALM_BODY_NAMES,
    PALM_FRAME_NAMES,
    POSITION_MAX_GATE_M,
    POSITION_P95_GATE_M,
    REFERENCE_PATH,
    REFERENCE_PROVENANCE,
    REFERENCE_SHA256,
    RIGHT_ARM_JOINT_NAMES,
    ROOT_HEIGHT_RANGE_M,
    ROOT_TILT_GATE_DEG,
    S2_02_CANDIDATE,
    S2_02_CANDIDATE_INDEX,
    S2_02_CONFIG,
    S2_02_CONFIG_SHA256,
    S2_02_DESIRED_PALM_QUATERNION_OBJECT_WXYZ,
    S2_02_FORMAL_RUN,
    S2_02_PALM_LOCAL_NORMAL_AXIS,
    S2_02_PELVIS_CONVERSION_SYMBOL,
    S2_02_RESULT_STATUS,
    S2_02_SOURCE_COMMIT,
    S2_02_SOURCE_SYMBOL,
    SETTLE_STEPS,
    STAGE,
    certified_object_local_targets,
    minimum_jerk_fraction,
    validate_reference_metadata,
)

LOWER_BODY_COMMAND = (0.0, 0.0, 0.0, 0.7)
LOWER_CHECKPOINT_SHA256 = "a0151975a757a33f0f5ed236d5616e2a98643abe2b36e50ac9ad01b6dd1f6d7e"
ENV: Any = None


def finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def tolist(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): tolist(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tolist(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def clamp_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value * torch.clamp(limit / norm.clamp_min(1.0e-9), max=1.0)


def root_metrics(quat: torch.Tensor) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))
    return roll, pitch, yaw, tilt


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
                ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", str(fps),
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
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
        self.env = env
        self.front_camera = env.scene["chest_prepose_front_camera"]
        self.side_camera = env.scene["chest_prepose_side_camera"]
        self.width = int(self.front_camera.cfg.width)
        self.height = int(self.front_camera.cfg.height)
        if (int(self.side_camera.cfg.width), int(self.side_camera.cfg.height)) != (self.width, self.height):
            raise RuntimeError("CAMERA_RESOLUTION_MISMATCH")
        self.front = VideoWriter(run_root / "chest_prepose_front.mp4", self.width, self.height)
        self.side = VideoWriter(run_root / "chest_prepose_side.mp4", self.width, self.height)
        self.font = ImageFont.load_default()
        self.counter = 0

    def overlay(self, rgb: np.ndarray, record: dict[str, Any]) -> Image.Image:
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image, mode="RGBA")
        lines = (
            f"phase={record['phase']}",
            f"frame={record['frame']}",
            f"left_pos_err_m={record['left_position_error_m']:.5f}",
            f"right_pos_err_m={record['right_position_error_m']:.5f}",
            f"left_ori_err_deg={record['left_orientation_error_deg']:.3f}",
            f"right_ori_err_deg={record['right_orientation_error_deg']:.3f}",
            f"torque_ratio={record['arm_torque_ratio_max']:.3f}",
            f"root_tilt_deg={record['root_tilt_deg']:.3f}",
            f"oscillation_detected={record['oscillation_detected']}",
        )
        h = 13
        draw.rectangle((4, 4, 315, 12 + h * len(lines)), fill=(0, 0, 0, 175))
        for i, line in enumerate(lines):
            draw.text((8, 7 + h * i), line, fill=(255, 255, 255, 255), font=self.font)
        return image

    def capture(self, record: dict[str, Any], *, force: bool = False, keyframe: str | None = None) -> None:
        front = self.overlay(camera_rgb(self.front_camera, self.width, self.height), record)
        side = self.overlay(camera_rgb(self.side_camera, self.width, self.height), record)
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
            "front": {"path": str(self.front.path), "size_bytes": self.front.path.stat().st_size, "sha256": sha256(self.front.path), "encoded_frames": self.front.frames},
            "side": {"path": str(self.side.path), "size_bytes": self.side.path.stat().st_size, "sha256": sha256(self.side.path), "encoded_frames": self.side.frames},
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
    failures = validate_reference_metadata(reference)
    if failures:
        raise RuntimeError("TARGET_REFERENCE_INVALID:" + ",".join(failures))
    source = reference.get("source", {})
    if source.get("controller_checkpoint_sha256") != LOWER_CHECKPOINT_SHA256:
        raise RuntimeError("TARGET_REFERENCE_LOWER_CHECKPOINT_SHA_MISMATCH")
    if not finite(reference):
        raise RuntimeError("TARGET_REFERENCE_NONFINITE")
    return reference


def target_pose_from_s2_02(
    reference: dict[str, Any],
    device: str,
    current_root_pos_w: torch.Tensor,
    current_root_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Convert the certified object-frame target into the *current* pelvis frame.

    The frozen reference stores robot and virtual-box states in the reference
    world frame.  Reusing those absolute coordinates would silently bake the
    tiny reset/randomization offset from the old run into this run.  First
    derive the certified box pose relative to the reference pelvis, then place
    that same virtual pose at the current reset pelvis and convert the palm
    targets back into the current pelvis frame.
    """
    local = torch.tensor(certified_object_local_targets(), dtype=torch.float32, device=device)
    reference_root = reference["robot_root_state_relative"].to(device=device, dtype=torch.float32)
    reference_box = reference["box_root_state_relative"].to(device=device, dtype=torch.float32)
    box_pos_b, box_quat_b = math_utils.subtract_frame_transforms(
        reference_root[:3].unsqueeze(0),
        reference_root[3:7].unsqueeze(0),
        reference_box[:3].unsqueeze(0),
        reference_box[3:7].unsqueeze(0),
    )
    current_box_pos_w, current_box_quat_w = math_utils.combine_frame_transforms(
        current_root_pos_w.unsqueeze(0),
        current_root_quat_w.unsqueeze(0),
        box_pos_b,
        box_quat_b,
    )
    object_q = torch.tensor(S2_02_DESIRED_PALM_QUATERNION_OBJECT_WXYZ, device=device).expand(2, 4)
    world_pos, world_quat = math_utils.combine_frame_transforms(
        current_box_pos_w.expand(2, 3), current_box_quat_w.expand(2, 4), local, object_q
    )
    target_pos, target_quat = math_utils.subtract_frame_transforms(
        current_root_pos_w.expand(2, 3), current_root_quat_w.expand(2, 4), world_pos, world_quat
    )
    return target_pos, target_quat, {
        "reference_box_pose_relative_to_reference_root": {
            "position_m": tolist(box_pos_b[0]),
            "quaternion_wxyz": tolist(box_quat_b[0]),
        },
        "current_root_pose_world": {
            "position_m": tolist(current_root_pos_w),
            "quaternion_wxyz": tolist(current_root_quat_w),
        },
        "current_virtual_box_pose_world": {
            "position_m": tolist(current_box_pos_w[0]),
            "quaternion_wxyz": tolist(current_box_quat_w[0]),
        },
    }


def recurrent_reset_audit(env: Any) -> dict[str, Any]:
    """Verify that each episode reset clears the frozen LSTM and action state."""

    lower = env.action_manager.get_term("lower_body_joint_pos")
    policy = getattr(lower, "_policy", None)
    hidden = getattr(policy, "hidden_state", None)
    cell = getattr(policy, "cell_state", None)
    previous = getattr(lower, "_previous_policy_actions", None)
    if not all(isinstance(value, torch.Tensor) for value in (hidden, cell, previous)):
        return {
            "reset_verified": False,
            "reason": "RECURRENT_STATE_FIELDS_MISSING",
        }
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


def usd_hand_audit(stage: Any, robot_path: str) -> dict[str, Any]:
    root = stage.GetPrimAtPath(robot_path)
    if not root.IsValid():
        raise RuntimeError(f"ROBOT_PRIM_MISSING:{robot_path}")
    prims = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if any(token in path.lower() for token in ("hand", "finger", "palm")):
            prims.append({"path": path, "type": prim.GetTypeName()})
    return {
        "hand_visual_type": "Unitree G1 Dex3-1 hand (runtime USD hand/finger prim audit)",
        "hardware_hand_match": "UNKNOWN",
        "hand_visual_prims": prims,
    }


def term_joint_ids(term: Any, count: int) -> set[int]:
    ids = getattr(term, "_joint_ids", None)
    if isinstance(ids, slice):
        return set(range(count))[ids]
    return set() if ids is None else {int(value) for value in ids}


def runtime_audit(env: Any, cfg: Any, stage: Any) -> dict[str, Any]:
    robot = env.scene["robot"]
    if "contact_forces" not in env.scene.keys():
        raise RuntimeError("FORBIDDEN_CONTACT_SENSOR_MISSING")
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
    if len(arm_ids) != 14:
        raise RuntimeError("ARM_JOINT_COUNT_INVALID")

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
        overlap[name] = [robot.joint_names[i] for i in sorted(ids & set(arm_ids))]
    arm_writers = {"left_hand_pose", "right_hand_pose"}
    overridden = any(bool(value) for name, value in overlap.items() if name not in arm_writers)

    lower = env.action_manager.get_term("lower_body_joint_pos")
    checkpoint_path = str(getattr(lower.cfg, "policy_path", getattr(lower, "checkpoint_path", "")))
    checkpoint_sha = getattr(lower, "checkpoint_sha256", None)
    if checkpoint_sha is None and checkpoint_path:
        checkpoint_sha = sha256(Path(checkpoint_path))
    if checkpoint_sha != LOWER_CHECKPOINT_SHA256:
        raise RuntimeError(f"LOWER_CHECKPOINT_SHA_MISMATCH:{checkpoint_sha}")
    source_pos = env.scene["hand_frames"].data.target_pos_source[0]
    source_quat = env.scene["hand_frames"].data.target_quat_source[0]
    if source_pos.shape != (2, 3) or source_quat.shape != (2, 4):
        raise RuntimeError(
            f"END_EFFECTOR_FRAME_INVALID:pose_shapes={tuple(source_pos.shape)}:{tuple(source_quat.shape)}"
        )
    usd_path = str(getattr(cfg.scene.robot.spawn, "usd_path", "METRIC_MISSING"))
    visual = usd_hand_audit(stage, "/World/envs/env_0/Robot")
    return {
        "asset": {
            "usd_path": usd_path,
            "urdf_path": "METRIC_MISSING",
            "robot_prim_path": "/World/envs/env_0/Robot",
        },
        "hand": visual,
        "wrist_body_names": list(wrist_names),
        "wrist_body_ids": [int(value) for value in wrist_ids],
        "end_effector_body_names": list(palm_names),
        "end_effector_body_ids": [int(value) for value in palm_ids],
        "palm_frame_names": frame_names,
        "palm_frame_paths": [
            "/World/envs/env_0/Robot/left_hand/left_hand_palm_link",
            "/World/envs/env_0/Robot/right_hand/right_hand_palm_link",
        ],
        "palm_pose_in_pelvis_at_reset": {
            "position_m": tolist(source_pos),
            "quaternion_wxyz": tolist(source_quat),
        },
        "palm_tcp_contract": {
            "tcp_body": "palm_link (no separate TCP prim)",
            "palm_collision_support_offset_m": float(
                S2_02_CANDIDATE["palm_collision_support_offset_m"]
            ),
            "local_normal_axis": list(S2_02_PALM_LOCAL_NORMAL_AXIS),
        },
        "arm_joint_names": list(arm_names),
        "arm_joint_ids": [int(value) for value in arm_ids],
        "left_arm_joint_names": list(left_names),
        "left_arm_joint_ids": [int(value) for value in left_ids],
        "right_arm_joint_names": list(right_names),
        "right_arm_joint_ids": [int(value) for value in right_ids],
        "action_terms": {
            "names": names,
            "dimensions": dims,
            "slices": slices,
            "arm_overlap_by_term": overlap,
            "lower_body_joint_names": list(getattr(lower, "_joint_names", ())),
            "lower_body_checkpoint_path": checkpoint_path,
            "lower_body_checkpoint_sha256": checkpoint_sha,
            "lower_body_command": list(LOWER_BODY_COMMAND),
            "upper_body_target_source": ["left_hand_pose", "right_hand_pose"],
            "final_actuator_target_api": "robot.data.joint_pos_target + set_joint_position_target",
        },
        "no_box_scene": True,
        "no_actor": True,
        "no_ppo": True,
        "no_planner": True,
        "no_base_command": True,
        "upper_body_target_overridden": overridden,
    }


def forbidden_contact(sensor: Any) -> tuple[bool | None, float | None, list[str]]:
    forces = getattr(getattr(sensor, "data", None), "net_forces_w", None)
    if forces is None:
        return None, None, []
    values = forces[0]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = torch.linalg.vector_norm(values, dim=-1)
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


def pose_command(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    desired_pos: torch.Tensor,
    desired_quat: torch.Tensor,
) -> torch.Tensor:
    pos_error, ori_error = math_utils.compute_pose_error(
        current_pos, current_quat, desired_pos, desired_quat, rot_error_type="axis_angle"
    )
    return torch.cat(
        (clamp_norm(pos_error, MAX_POSITION_COMMAND_M), clamp_norm(ori_error, MAX_ORIENTATION_COMMAND_RAD)),
        dim=-1,
    )


def desired_pose(
    baseline_pos: torch.Tensor,
    baseline_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    fraction: float,
    active: tuple[bool, bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    u = min(1.0, max(0.0, float(fraction)))
    pos = baseline_pos.clone()
    quat = baseline_quat.clone()
    for side in range(2):
        if active[side]:
            pos[side] = baseline_pos[side] + u * (target_pos[side] - baseline_pos[side])
            quat[side] = math_utils.quat_slerp(baseline_quat[side], target_quat[side], tau=u)
    return pos, quat


def make_record(
    env: Any,
    phase: str,
    frame: int,
    desired_pos: torch.Tensor,
    desired_quat: torch.Tensor,
    previous_target: torch.Tensor | None,
    oscillation: bool = False,
) -> tuple[dict[str, Any], torch.Tensor]:
    robot = env.scene["robot"]
    palms = env.scene["hand_frames"]
    arm_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
    actual_pos = palms.data.target_pos_source[0].detach().clone()
    actual_quat = palms.data.target_quat_source[0].detach().clone()
    pos_error, ori_error = math_utils.compute_pose_error(
        actual_pos, actual_quat, desired_pos, desired_quat, rot_error_type="axis_angle"
    )
    target = robot.data.joint_pos_target[0, arm_ids].detach().clone()
    actual_joint = robot.data.joint_pos[0, arm_ids].detach().clone()
    velocity = robot.data.joint_vel[0, arm_ids].detach().clone()
    limits = robot.data.joint_pos_limits[0, arm_ids]
    margin = torch.minimum(actual_joint - limits[:, 0], limits[:, 1] - actual_joint).min()
    effort = robot.data.joint_effort_limits[0, arm_ids].abs().clamp_min(1.0e-6)
    torque_ratio = (robot.data.applied_torque[0, arm_ids].abs() / effort).max()
    roll, pitch, yaw, tilt = root_metrics(robot.data.root_link_quat_w[0])
    collision, force_max, bodies = forbidden_contact(env.scene["contact_forces"])
    rate = 0.0 if previous_target is None else float((target - previous_target).abs().max() / CONTROL_DT_S)
    finite_state = (
        bool(torch.isfinite(actual_pos).all())
        and bool(torch.isfinite(actual_quat).all())
        and bool(torch.isfinite(target).all())
        and bool(torch.isfinite(actual_joint).all())
        and bool(torch.isfinite(velocity).all())
        and bool(torch.isfinite(robot.data.root_link_pos_w[0]).all())
        and bool(torch.isfinite(robot.data.applied_torque[0, arm_ids]).all())
        and collision is not None
    )
    record = {
        "frame": int(frame),
        "time_s": float(frame) * CONTROL_DT_S,
        "phase": phase,
        "finite": finite_state,
        "desired_left_palm_pose_pelvis": {"position_m": tolist(desired_pos[0]), "quaternion_wxyz": tolist(desired_quat[0])},
        "desired_right_palm_pose_pelvis": {"position_m": tolist(desired_pos[1]), "quaternion_wxyz": tolist(desired_quat[1])},
        "actual_left_palm_pose_pelvis": {"position_m": tolist(actual_pos[0]), "quaternion_wxyz": tolist(actual_quat[0])},
        "actual_right_palm_pose_pelvis": {"position_m": tolist(actual_pos[1]), "quaternion_wxyz": tolist(actual_quat[1])},
        "left_position_error_m": float(torch.linalg.vector_norm(pos_error[0])),
        "right_position_error_m": float(torch.linalg.vector_norm(pos_error[1])),
        "left_position_error_xyz_m": tolist(pos_error[0]),
        "right_position_error_xyz_m": tolist(pos_error[1]),
        "left_orientation_error_deg": math.degrees(float(torch.linalg.vector_norm(ori_error[0]))),
        "right_orientation_error_deg": math.degrees(float(torch.linalg.vector_norm(ori_error[1]))),
        "desired_upper_body_joints_rad": tolist(target),
        "actual_upper_body_joints_rad": tolist(actual_joint),
        "final_actuator_target_rad": tolist(target),
        "upper_joint_velocity_rad_s": tolist(velocity),
        "upper_joint_velocity_max_rad_s": float(velocity.abs().max()),
        "action_rate_max_rad_s": rate,
        "action_rate_source": "DifferentialIK joint target delta / control_dt",
        "arm_torque_ratio_max": float(torque_ratio),
        "minimum_joint_limit_margin_rad": float(margin),
        "root_height_m": float(robot.data.root_link_pos_w[0, 2]),
        "root_roll_deg": math.degrees(roll),
        "root_pitch_deg": math.degrees(pitch),
        "root_yaw_deg": math.degrees(yaw),
        "root_tilt_deg": math.degrees(tilt),
        "forbidden_collision": bool(collision) if collision is not None else None,
        "forbidden_contact_force_max_n": force_max,
        "forbidden_contact_bodies": bodies,
        "action_source": {
            "left_arm": "action_manager.left_hand_pose:DifferentialIK:DLS",
            "right_arm": "action_manager.right_hand_pose:DifferentialIK:DLS",
            "lower_body": "action_manager.lower_body_joint_pos:frozen_recurrent_student",
            "base": "fixed_zero_command",
        },
        "upper_body_target_overridden": False,
        "joint_order_swapped": False,
        "ik_frame_error": False,
        "pose_delta_accumulation_detected": False,
        "oscillation_detected": oscillation,
    }
    return record, target


def oscillation_detected(records: list[dict[str, Any]], phase: str) -> bool:
    hold = [item for item in records if item["phase"] == phase]
    if len(hold) < 12:
        return False
    detected = False
    for key in ("actual_left_palm_pose_pelvis", "actual_right_palm_pose_pelvis"):
        points = np.asarray([item[key]["position_m"] for item in hold], dtype=float)
        velocity = np.diff(points, axis=0) / CONTROL_DT_S
        for axis in range(3):
            values = velocity[:, axis]
            values = values[np.abs(values) > 1.0e-4]
            reversals = int(np.sum(np.sign(values[1:]) * np.sign(values[:-1]) < 0)) if len(values) > 1 else 0
            amplitude = float(points[:, axis].max() - points[:, axis].min())
            if reversals >= 8 and amplitude > 0.003:
                detected = True
    return detected


def set_camera_views(env: Any) -> None:
    env.scene["chest_prepose_front_camera"].set_world_poses_from_view(
        torch.tensor([[1.8, -2.2, 1.35]], device=env.device),
        torch.tensor([[0.0, 0.0, 0.82]], device=env.device),
    )
    env.scene["chest_prepose_side_camera"].set_world_poses_from_view(
        torch.tensor([[2.25, 0.05, 1.08]], device=env.device),
        torch.tensor([[0.0, 0.0, 0.80]], device=env.device),
    )


def run_episode(
    env: Any,
    *,
    label: str,
    active: tuple[bool, bool],
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    formal: bool = False,
    video: EvidenceVideo | None = None,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=42)
    if not finite(observation):
        raise RuntimeError(f"OBSERVATION_NONFINITE_AFTER_RESET:{label}")
    reset_audit = recurrent_reset_audit(env)
    if not reset_audit["reset_verified"]:
        raise RuntimeError(f"RECURRENT_RESET_INVALID:{label}:{reset_audit}")
    palms = env.scene["hand_frames"]
    baseline_pos = palms.data.target_pos_source[0].detach().clone()
    baseline_quat = palms.data.target_quat_source[0].detach().clone()
    names = list(env.action_manager.active_terms)
    dims = [int(value) for value in env.action_manager.action_term_dim]
    slices: dict[str, slice] = {}
    cursor = 0
    for name, dim in zip(names, dims, strict=True):
        slices[name] = slice(cursor, cursor + dim)
        cursor += dim
    required_terms = {"left_hand_pose", "right_hand_pose", "lower_body_joint_pos"}
    if not required_terms.issubset(slices):
        raise RuntimeError(f"ACTION_CONTRACT_INVALID:{names}")
    actions = torch.zeros((1, cursor), dtype=torch.float32, device=env.device)
    lower_command = torch.tensor(LOWER_BODY_COMMAND, dtype=torch.float32, device=env.device)
    records: list[dict[str, Any]] = []
    previous_target: torch.Tensor | None = None
    early_done = False

    if video is not None:
        initial, initial_target = make_record(
            env, "INITIAL", 0, baseline_pos, baseline_quat, None
        )
        initial["desired_upper_body_joints_rad"] = tolist(initial_target)
        video.capture(initial, force=True, keyframe="initial")

    def one_step(phase: str, fraction: float, frame: int) -> None:
        nonlocal previous_target, early_done
        desired_pos, desired_quat = desired_pose(
            baseline_pos, baseline_quat, target_pos, target_quat, fraction, active
        )
        current_pos = palms.data.target_pos_source[0]
        current_quat = palms.data.target_quat_source[0]
        actions.zero_()
        actions[0, slices["left_hand_pose"]] = pose_command(
            current_pos[0:1], current_quat[0:1], desired_pos[0:1], desired_quat[0:1]
        )[0]
        actions[0, slices["right_hand_pose"]] = pose_command(
            current_pos[1:2], current_quat[1:2], desired_pos[1:2], desired_quat[1:2]
        )[0]
        actions[0, slices["lower_body_joint_pos"]] = lower_command
        # This is a ManagerBasedEnv (not ManagerBasedRLEnv), so its step
        # contract is exactly ``(observation, extras)``.  Keeping this
        # explicit prevents an accidental dependency on an RL termination
        # manager, which is intentionally absent from the no-box slice.
        observation, extras = env.step(actions)
        del extras
        if not finite(observation):
            raise RuntimeError(f"NONFINITE_RUNTIME_STATE:{label}:{frame}")
        record, previous_target = make_record(
            env, phase, frame, desired_pos, desired_quat, previous_target
        )
        records.append(record)
        if (
            not record["finite"]
            or record["forbidden_collision"] is True
            or record["root_height_m"] < ROOT_HEIGHT_RANGE_M[0]
            or record["root_height_m"] > ROOT_HEIGHT_RANGE_M[1]
            or record["root_tilt_deg"] > ROOT_TILT_GATE_DEG
            or record["minimum_joint_limit_margin_rad"] < 0.10
            or record["arm_torque_ratio_max"] > 1.001
        ):
            early_done = True
        if video is not None:
            video.capture(record, force=False)
        if frame % 25 == 0 or frame == 1:
            print(f"PHASE={phase} step={frame}", flush=True)

    for frame in range(1, SETTLE_STEPS + 1):
        one_step("FORMAL_SETTLE" if formal else "SETTLE", 0.0, frame)
        if early_done:
            break
    if not early_done:
        for index in range(1, MOVE_STEPS + 1):
            frame = SETTLE_STEPS + index
            one_step(
                "FORMAL_MOVE" if formal else "MOVE",
                minimum_jerk_fraction(index, MOVE_STEPS),
                frame,
            )
            if early_done:
                break
    hold_count = HOLD_STEPS if formal else 100
    if not early_done:
        for index in range(1, hold_count + 1):
            frame = SETTLE_STEPS + MOVE_STEPS + index
            one_step("FORMAL_HOLD" if formal else "HOLD", 1.0, frame)
            if video is not None and index == 1:
                video.capture(records[-1], force=True, keyframe="prepose")
            if early_done:
                break

    hold_phase = "FORMAL_HOLD" if formal else "HOLD"
    oscillation = oscillation_detected(records, hold_phase)
    for item in records:
        item["oscillation_detected"] = oscillation
    if video is not None and records:
        video.capture(records[-1], force=True, keyframe="terminal")

    hold_records = [item for item in records if item["phase"] == hold_phase]
    left_pos = [float(item["left_position_error_m"]) for item in hold_records]
    right_pos = [float(item["right_position_error_m"]) for item in hold_records]
    left_ori = [float(item["left_orientation_error_deg"]) for item in hold_records]
    right_ori = [float(item["right_orientation_error_deg"]) for item in hold_records]
    safety = all(
        bool(item["finite"])
        and item["forbidden_collision"] is False
        and ROOT_HEIGHT_RANGE_M[0] <= float(item["root_height_m"]) <= ROOT_HEIGHT_RANGE_M[1]
        and float(item["root_tilt_deg"]) <= ROOT_TILT_GATE_DEG
        and float(item["minimum_joint_limit_margin_rad"]) >= 0.10
        and float(item["arm_torque_ratio_max"]) <= 1.001
        for item in records
    )
    tracking = bool(
        hold_records
        and percentile(left_pos, 95) <= POSITION_P95_GATE_M
        and percentile(right_pos, 95) <= POSITION_P95_GATE_M
        and max(left_pos, default=float("inf")) <= POSITION_MAX_GATE_M
        and max(right_pos, default=float("inf")) <= POSITION_MAX_GATE_M
        and percentile(left_ori, 95) <= 5.0
        and percentile(right_ori, 95) <= 5.0
        and max(left_ori, default=float("inf")) <= 8.0
        and max(right_ori, default=float("inf")) <= 8.0
        and not oscillation
    )
    return {
        "label": label,
        "active_arms": {"left": active[0], "right": active[1]},
        "baseline_palm_pose_pelvis": {"position_m": tolist(baseline_pos), "quaternion_wxyz": tolist(baseline_quat)},
        "target_palm_pose_pelvis": {"position_m": tolist(target_pos), "quaternion_wxyz": tolist(target_quat)},
        "observed_control_frames": len(records),
        "expected_control_frames": SETTLE_STEPS + MOVE_STEPS + hold_count,
        "hold_frames": len(hold_records),
        "hold_seconds": len(hold_records) * CONTROL_DT_S,
        "recurrent_reset": reset_audit,
        "early_done": early_done,
        "oscillation_detected": oscillation,
        "tracking_ok": tracking,
        "safety_ok": safety,
        "position_error_p95_m": {"left": percentile(left_pos, 95), "right": percentile(right_pos, 95)},
        "position_error_max_m": {"left": max(left_pos, default=float("nan")), "right": max(right_pos, default=float("nan"))},
        "orientation_error_p95_deg": {"left": percentile(left_ori, 95), "right": percentile(right_ori, 95)},
        "orientation_error_max_deg": {"left": max(left_ori, default=float("nan")), "right": max(right_ori, default=float("nan"))},
        "records": records,
    }


def classify(formal: dict[str, Any], runtime: dict[str, Any], probes: list[dict[str, Any]]) -> tuple[str, str]:
    if runtime["upper_body_target_overridden"]:
        return "FAIL", "UPPER_BODY_TARGET_OVERRIDDEN"
    if formal["early_done"] or not formal["safety_ok"]:
        return "FAIL", "SAFETY_GATE_TRIGGERED"
    if formal["hold_seconds"] < HOLD_SECONDS:
        return "INVALID", "INCOMPLETE_TRACE"
    if not formal["tracking_ok"]:
        return "FAIL", "CHEST_PREPOSE_TRACKING_UNSTABLE"
    for probe in probes:
        if probe["early_done"]:
            return "FAIL", "SAFETY_GATE_TRIGGERED"
        if not probe["tracking_ok"]:
            return "FAIL", "CHEST_PREPOSE_TRACKING_UNSTABLE"
    return "PASS", "CHEST_PREPOSE_TRACKING_COMPLETE"


def main() -> int:
    global ENV
    reference_path = (args.reference or REFERENCE_PATH).resolve()
    write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": "REFERENCE_AUDIT"})
    reference = load_reference(reference_path)
    config_path = (Path.cwd() / S2_02_CONFIG).resolve()
    config_sha = sha256(config_path)
    if config_sha != S2_02_CONFIG_SHA256:
        raise RuntimeError(f"S2_02_CONFIG_SHA_MISMATCH:{config_sha}")

    cfg = S203TChestPreposeEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    # ManagerBasedEnv is the no-box manager environment and does not accept a
    # Gym ``render_mode`` constructor argument.  Cameras in the scene provide
    # the rgb-array evidence path when ``--enable_cameras`` is passed to the
    # AppLauncher.
    ENV = ManagerBasedEnv(cfg=cfg)
    write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": "ENVIRONMENT_CREATED"})
    set_camera_views(ENV)
    runtime = runtime_audit(ENV, cfg, get_current_stage())
    # Establish the deterministic reset before resolving the virtual-box
    # geometry in the current pelvis frame.
    initial_observation, _ = ENV.reset(seed=args.seed)
    if not finite(initial_observation):
        raise RuntimeError("OBSERVATION_NONFINITE_AFTER_INITIAL_RESET")
    current_robot = ENV.scene["robot"]
    target_pos, target_quat, target_conversion = target_pose_from_s2_02(
        reference,
        ENV.device,
        current_robot.data.root_link_pos_w[0].detach().clone(),
        current_robot.data.root_link_quat_w[0].detach().clone(),
    )
    if not bool(torch.isfinite(target_pos).all() and torch.isfinite(target_quat).all()):
        raise RuntimeError("TARGET_POSE_NONFINITE")

    write_json(
        RUN / "resolved_config.json",
        {
            "schema_version": 1,
            "stage": STAGE,
            "seed": args.seed,
            "control_dt_s": CONTROL_DT_S,
            "num_envs": 1,
            "render_mode": "rgb_array",
            "enable_cameras": True,
            "no_box": True,
            "no_actor": True,
            "no_ppo": True,
            "no_planner": True,
            "reference": {"path": str(reference_path), "sha256": sha256(reference_path), "provenance": REFERENCE_PROVENANCE},
            "s2_02": {
                "formal_run": S2_02_FORMAL_RUN,
                "source_commit": S2_02_SOURCE_COMMIT,
                "result_status": S2_02_RESULT_STATUS,
                "candidate_index": S2_02_CANDIDATE_INDEX,
                "config_path": str(S2_02_CONFIG),
                "config_sha256": config_sha,
                "target_source": S2_02_SOURCE_SYMBOL,
                "pelvis_conversion_source": S2_02_PELVIS_CONVERSION_SYMBOL,
                "conversion_audit": target_conversion,
                "candidate": S2_02_CANDIDATE,
                "object_local_targets_m": certified_object_local_targets(),
                "desired_quaternion_object_wxyz": list(S2_02_DESIRED_PALM_QUATERNION_OBJECT_WXYZ),
                "target_pose_pelvis_m": tolist(target_pos),
                "target_quaternion_pelvis_wxyz": tolist(target_quat),
            },
            "runtime": runtime,
            "thresholds": {
                "position_p95_m": POSITION_P95_GATE_M,
                "position_max_m": POSITION_MAX_GATE_M,
                "orientation_p95_deg": 5.0,
                "orientation_max_deg": 8.0,
                "joint_margin_rad": 0.10,
                "torque_ratio": 1.001,
                "root_height_range_m": list(ROOT_HEIGHT_RANGE_M),
                "root_tilt_deg": ROOT_TILT_GATE_DEG,
                "forbidden_force_n": FORBIDDEN_FORCE_GATE_N,
            },
        },
    )

    probes: list[dict[str, Any]] = []
    for label, active in (("LEFT_ONLY", (True, False)), ("RIGHT_ONLY", (False, True)), ("SYNCHRONIZED_MOVE", (True, True))):
        write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": label})
        probe = run_episode(
            ENV,
            label=label,
            active=active,
            target_pos=target_pos,
            target_quat=target_quat,
        )
        write_json(
            RUN / f"{label.lower()}_trace.json",
            {"stage": STAGE, "label": label, "records": probe["records"]},
        )
        probes.append(probe)

    write_json(RUN / "runner_status.json", {"status": "RUNNING", "phase": "FORMAL_SYNC_HOLD"})
    global RECORDER
    RECORDER = EvidenceVideo(RUN, ENV)
    formal = run_episode(
        ENV,
        label="FORMAL_SYNCHRONIZED_HOLD",
        active=(True, True),
        target_pos=target_pos,
        target_quat=target_quat,
        formal=True,
        video=RECORDER,
    )
    videos = RECORDER.close()
    status, reason = classify(formal, runtime, probes)
    result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": status,
        "primary_reason": reason,
        "reference": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "provenance": REFERENCE_PROVENANCE,
            "direct_s2_02_joint_reference": "METRIC_MISSING",
        },
        "runtime_audit": runtime,
        "probe_results": [{key: value for key, value in item.items() if key != "records"} for item in probes],
        "formal": {key: value for key, value in formal.items() if key != "records"},
        "formal_trace": formal["records"],
        "trajectory": {
            "control_mode": "DIFFERENTIAL_IK",
            "target_source": S2_02_SOURCE_SYMBOL,
            "pelvis_conversion_source": S2_02_PELVIS_CONVERSION_SYMBOL,
            "minimum_jerk": True,
            "pose_delta_not_accumulated": True,
            "settle_steps": SETTLE_STEPS,
            "move_steps": MOVE_STEPS,
            "hold_steps": HOLD_STEPS,
            "hold_seconds": HOLD_SECONDS,
        },
        "videos": videos,
        "scientific_contract": {
            "box_used": False,
            "contact_test_started": False,
            "base_walking_started": False,
            "ppo_started": False,
            "checkpoint_training_started": False,
            "falcon_started": False,
            "planner_started": False,
        },
        "authoritative_complete_before_teardown": True,
        "observed_control_frames": formal["observed_control_frames"],
        "expected_control_frames": formal["expected_control_frames"],
        "hold_seconds": formal["hold_seconds"],
        "recurrent_reset": formal["recurrent_reset"],
    }
    write_json(RUN / "result.json", result)
    write_json(
        RUN / "summary.json",
        {
            "stage": STAGE,
            "status": status,
            "primary_reason": reason,
            "observed_control_frames": formal["observed_control_frames"],
            "expected_control_frames": formal["expected_control_frames"],
            "hold_seconds": formal["hold_seconds"],
            "left_position_error_p95_m": formal["position_error_p95_m"]["left"],
            "right_position_error_p95_m": formal["position_error_p95_m"]["right"],
            "left_position_error_max_m": formal["position_error_max_m"]["left"],
            "right_position_error_max_m": formal["position_error_max_m"]["right"],
            "oscillation_detected": formal["oscillation_detected"],
            "root_stability": "PASS" if formal["safety_ok"] else "FAIL",
            "safety_gates": "PASS" if formal["safety_ok"] else "FAIL",
            "video_paths": videos,
        },
    )
    write_json(RUN / "visualization_status.json", {"visualization_status": "PASS", "scientific_result_status": status})
    write_json(RUN / "runner_status.json", {"status": "COMPLETE", "phase": "RESULT_WRITTEN", "primary_reason": reason})
    print(f"CHEST_PREPOSE_STATUS={status}", flush=True)
    print(f"PRIMARY_REASON={reason}", flush=True)
    print(f"OBSERVED_CONTROL_FRAMES={formal['observed_control_frames']}", flush=True)
    print(f"HOLD_SECONDS={formal['hold_seconds']}", flush=True)
    return 0


RECORDER: EvidenceVideo | None = None
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
    write_json(RUN / "result.json", payload)
    write_json(RUN / "runner_status.json", {"status": "INVALID", "phase": "EXCEPTION", "primary_reason": "IMPLEMENTATION_EXCEPTION"})
    print("CHEST_PREPOSE_STATUS=INVALID", flush=True)
    print("PRIMARY_REASON=IMPLEMENTATION_EXCEPTION", flush=True)
    rc = 1
finally:
    if RECORDER is not None:
        try:
            RECORDER.close()
        except Exception:
            pass
    if ENV is not None:
        try:
            ENV.close()
        except Exception:
            pass
    try:
        simulation_app.close()
    except Exception:
        pass

raise SystemExit(rc)
