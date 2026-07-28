"""Off-screen two-view recorder for S2-03T deterministic evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import isaaclab.utils.math as math_utils


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class _RawVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:  # pragma: no cover - host dependency
                raise RuntimeError(f"FFMPEG_NOT_FOUND:{exc!r}") from exc
        self.path = path
        self.frame_count = 0
        self._process = subprocess.Popen(
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

    def write(self, rgb: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("FFMPEG_STDIN_CLOSED")
        self._process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.frame_count += 1

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
            self._process.stdin = None
        stderr = b"" if self._process.stderr is None else self._process.stderr.read()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFMPEG_FAILED:{return_code}:{stderr.decode(errors='replace')[-2000:]}")
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError(f"VIDEO_MISSING_OR_EMPTY:{self.path}")


class VisualEvidenceRecorder:
    """Record videos, keyframes, kinematic snapshots, and a compact manifest."""

    def __init__(
        self,
        run_root: Path,
        env: Any,
        *,
        controller_id: str,
        checkpoint_path: str | None,
        checkpoint_sha256: str | None,
        checkpoint_iteration: int | None,
        seed: int,
        reference_path: str,
        reference_sha256: str,
        frame_stride: int = 2,
        fps: int = 25,
    ) -> None:
        if frame_stride < 1:
            raise ValueError("visual frame stride must be positive")
        self.run_root = run_root
        self.env = env
        self.controller_id = controller_id
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_iteration = checkpoint_iteration
        self.seed = seed
        self.reference_path = reference_path
        self.reference_sha256 = reference_sha256
        self.frame_stride = frame_stride
        self.fps = fps
        self.front_camera = env.scene["audit_camera"]
        self.side_camera = env.scene["side_camera"]
        palm_frame_names = list(env.scene["hand_frames"].data.target_frame_names)
        if palm_frame_names != ["left_hand_palm", "right_hand_palm"]:
            raise RuntimeError(f"PALM_FRAME_ORDER_MISMATCH:{palm_frame_names}")

        self.height = int(self.front_camera.cfg.height)
        self.width = int(self.front_camera.cfg.width)
        if (int(self.side_camera.cfg.height), int(self.side_camera.cfg.width)) != (
            self.height,
            self.width,
        ):
            raise RuntimeError("VISUAL_CAMERA_RESOLUTION_MISMATCH")
        self.front_video_path = run_root / "video_three_quarter.mp4"
        self.side_video_path = run_root / "video_side.mp4"
        self.front_writer = _RawVideoWriter(self.front_video_path, self.width, self.height, fps)
        self.side_writer = _RawVideoWriter(self.side_video_path, self.width, self.height, fps)
        self.font = ImageFont.load_default()
        self.finalized = False
        self.observed_frame_count = 0
        self.last_observed_frame: int | None = None
        self.minimum_mean_gap_m = float("inf")
        self.minimum_mean_gap_frame: int | None = None
        self.minimum_left_gap_m = float("inf")
        self.minimum_left_gap_frame: int | None = None
        self.minimum_right_gap_m = float("inf")
        self.minimum_right_gap_frame: int | None = None
        self.maximum_force_n = [0.0, 0.0]
        self.contact_seen = [False, False]
        self.maximum_abs_residual_rad = 0.0
        self.maximum_abs_residual_per_joint_rad = [0.0] * 14
        self.maximum_abs_normalized_action_per_joint = [0.0] * 14
        self.maximum_original_commanded_normal_displacement_m = 0.0
        self.precontact_palm_world_position: torch.Tensor | None = None
        self.maximum_actual_palm_displacement_m = [0.0, 0.0]
        self.precontact_palm_box_position: torch.Tensor | None = None
        self.maximum_actual_palm_box_displacement_m = [0.0, 0.0]
        self.maximum_actual_palm_normal_closure_m = [0.0, 0.0]
        self.minimum_root_height_m = float("inf")
        self.maximum_root_tilt_deg = 0.0
        self.keyframe_snapshots: dict[str, dict[str, Any]] = {}
        self._minimum_front: Image.Image | None = None
        self._minimum_side: Image.Image | None = None

    def set_original_commanded_normal_displacement(self, value_m: float) -> None:
        self.maximum_original_commanded_normal_displacement_m = max(
            self.maximum_original_commanded_normal_displacement_m, abs(float(value_m))
        )

    def mark_precontact(self, record: dict[str, Any]) -> None:
        palm_pos_w, _ = self._palm_world_pose()
        self.precontact_palm_world_position = palm_pos_w.detach().clone()
        palm_pos_o, _ = self._palm_box_pose()
        self.precontact_palm_box_position = palm_pos_o.detach().clone()
        self._save_keyframe("precontact", record)

    def observe(
        self,
        record: dict[str, Any],
        *,
        keyframe: str | None = None,
        actor_phase: bool = False,
        force_video_frame: bool = False,
    ) -> None:
        frame = int(record["frame"])
        if self.last_observed_frame is not None and frame <= self.last_observed_frame:
            raise RuntimeError(f"VISUAL_FRAME_ORDER_INVALID:{self.last_observed_frame}:{frame}")
        self.last_observed_frame = frame

        self.observed_frame_count = max(self.observed_frame_count, frame + 1)
        front = self._overlay(self._camera_rgb(self.front_camera), record)
        side = self._overlay(self._camera_rgb(self.side_camera), record)
        if frame % self.frame_stride == 0 or force_video_frame:
            self.front_writer.write(np.asarray(front, dtype=np.uint8))
            self.side_writer.write(np.asarray(side, dtype=np.uint8))
        if keyframe is not None:
            self._save_images(keyframe, front, side)
            self.keyframe_snapshots[keyframe] = self._snapshot(record)
        if actor_phase:
            self._update_actor_metrics(record, front, side)

    def finalize(
        self,
        *,
        records: list[dict[str, Any]],
        terminal_state: str,
        failure_reason: str | None,
        result_primary_reason: str,
        installed_reference_max_abs_diff: float,
        reference_replay_max_abs_diff: float,
    ) -> dict[str, Any]:
        if self.finalized:
            raise RuntimeError("VISUAL_RECORDER_ALREADY_FINALIZED")
        if not records:
            raise RuntimeError("VISUAL_EVIDENCE_HAS_NO_RECORDS")
        terminal_record = records[-1]
        if "terminal" not in self.keyframe_snapshots:
            raise RuntimeError("TERMINAL_PRE_RESET_FRAME_MISSING")
        if self._minimum_front is None or self._minimum_side is None:
            raise RuntimeError("MINIMUM_GAP_FRAME_MISSING")
        self._minimum_front.save(self.run_root / "frame_minimum_gap.png")
        self._minimum_side.save(self.run_root / "frame_minimum_gap_side.png")
        self.front_writer.close()
        self.side_writer.close()
        video_manifest = {
            "three_quarter_front": {
                "path": str(self.front_video_path),
                "sha256": _sha256(self.front_video_path),
                "size_bytes": self.front_video_path.stat().st_size,
                "encoded_frames": self.front_writer.frame_count,
            },
            "side_view": {
                "path": str(self.side_video_path),
                "sha256": _sha256(self.side_video_path),
                "size_bytes": self.side_video_path.stat().st_size,
                "encoded_frames": self.side_writer.frame_count,
            },
        }
        image_paths = {
            "initial": self.run_root / "frame_initial.png",
            "initial_side": self.run_root / "frame_initial_side.png",
            "precontact": self.run_root / "frame_precontact.png",
            "precontact_side": self.run_root / "frame_precontact_side.png",
            "minimum_gap": self.run_root / "frame_minimum_gap.png",
            "minimum_gap_side": self.run_root / "frame_minimum_gap_side.png",
            "terminal": self.run_root / "frame_terminal.png",
            "terminal_side": self.run_root / "frame_terminal_side.png",
        }
        missing_images = [str(path) for path in image_paths.values() if not path.is_file() or path.stat().st_size == 0]
        if missing_images:
            raise RuntimeError(f"REQUIRED_VISUAL_IMAGES_MISSING:{missing_images}")
        image_manifest = {
            name: {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for name, path in image_paths.items()
        }

        payload = {
            "schema_version": 1,
            "stage": "S2-03T",
            "package": "S2_03T_VISUAL_EVIDENCE_PACKAGE",
            "visualization_status": "PASS",
            "visualization_primary_reason": "ALL_REQUIRED_VISUAL_EVIDENCE_READY",
            "scientific_result_status": terminal_state,
            "scientific_primary_reason": result_primary_reason,
            "scientific_failure_reason": failure_reason,
            "scientific_authority": "TRACE_SENSOR_EVALUATOR_ONLY",
            "controller_id": self.controller_id,
            "checkpoint": {
                "path": self.checkpoint_path,
                "sha256": self.checkpoint_sha256,
                "iteration": self.checkpoint_iteration,
            },
            "seed": self.seed,
            "reference": {
                "path": self.reference_path,
                "sha256": self.reference_sha256,
                "installed_max_abs_diff": installed_reference_max_abs_diff,
                "camera_enabled_replay_max_abs_diff_diagnostic_only": reference_replay_max_abs_diff,
            },
            "execution": {
                "num_envs": 1,
                "episode_count": 1,
                "auto_reset_accumulation": False,
                "box_push_commanded": False,
                "planner_started": False,
                "render_mode": "rgb_array",
                "enable_cameras": True,
                "resolution": [self.width, self.height],
                "video_fps": self.fps,
                "control_frame_stride": self.frame_stride,
                "terminal_capture": "PRE_AUTO_RESET_PHYSICS_FRAME",
                "terminal_frame_is_synthetic_aggregate": False,
            },
            "observed_frames": len(records),
            "minimum_gap": {
                "left_m": self.minimum_left_gap_m,
                "left_frame": self.minimum_left_gap_frame,
                "right_m": self.minimum_right_gap_m,
                "right_frame": self.minimum_right_gap_frame,
                "mean_m": self.minimum_mean_gap_m,
                "mean_frame": self.minimum_mean_gap_frame,
            },
            "contact": {
                "left_seen": self.contact_seen[0],
                "right_seen": self.contact_seen[1],
                "left_force_maximum_n": self.maximum_force_n[0],
                "right_force_maximum_n": self.maximum_force_n[1],
            },
            "action_and_motion": {
                "maximum_abs_commanded_residual_rad": self.maximum_abs_residual_rad,
                "maximum_abs_commanded_residual_per_joint_rad": self.maximum_abs_residual_per_joint_rad,
                "maximum_abs_normalized_action_per_joint": self.maximum_abs_normalized_action_per_joint,
                "maximum_original_commanded_normal_displacement_m": self.maximum_original_commanded_normal_displacement_m,
                "maximum_actual_palm_displacement_m": self.maximum_actual_palm_displacement_m,
                "maximum_actual_palm_box_displacement_m": self.maximum_actual_palm_box_displacement_m,
                "maximum_actual_palm_normal_closure_m": self.maximum_actual_palm_normal_closure_m,
            },
            "robot_stability": {
                "minimum_root_height_m": self.minimum_root_height_m,
                "maximum_root_tilt_deg": self.maximum_root_tilt_deg,
            },
            "terminal": {
                "state": terminal_state,
                "failure_reason": failure_reason,
                "left_contact": bool(terminal_record["left_contact"]),
                "right_contact": bool(terminal_record["right_contact"]),
                "left_force_n": float(terminal_record["left_force_n"]),
                "right_force_n": float(terminal_record["right_force_n"]),
            },
            "keyframe_snapshots": self.keyframe_snapshots,
            "videos": video_manifest,
            "required_images": image_manifest,
        }
        _write_json(self.run_root / "visual_evidence.json", payload)
        _write_json(
            self.run_root / "visualization_status.json",
            {
                "visualization_status": "PASS",
                "visualization_primary_reason": "ALL_REQUIRED_VISUAL_EVIDENCE_READY",
                "scientific_result_status": terminal_state,
            },
        )
        self.finalized = True
        return payload

    def close_incomplete(self, reason: str = "VISUAL_RECORDER_INCOMPLETE") -> None:
        if self.finalized:
            return
        errors = []
        for writer in (self.front_writer, self.side_writer):
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(repr(exc))
        self.finalized = True
        payload = {
            "visualization_status": "INVALID",
            "visualization_primary_reason": reason,
            "scientific_result_unchanged": True,
            "errors": errors,
        }
        _write_json(self.run_root / "visual_evidence_invalid.json", payload)
        _write_json(self.run_root / "visualization_status.json", payload)

    def _camera_rgb(self, camera: Any) -> np.ndarray:
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb, 0.0, 1.0) * 255.0
        rgb = rgb.astype(np.uint8, copy=False)
        if rgb.shape != (self.height, self.width, 3):
            raise RuntimeError(f"CAMERA_RGB_SHAPE_MISMATCH:{rgb.shape}")
        return rgb

    def _overlay(self, rgb: np.ndarray, record: dict[str, Any]) -> Image.Image:
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image, mode="RGBA")
        lines = (
            f"frame={int(record['frame'])}",
            f"left_gap_m={float(record['left_actual_surface_gap_m']):.5f}",
            f"right_gap_m={float(record['right_actual_surface_gap_m']):.5f}",
            f"left_contact={bool(record['left_contact'])}",
            f"right_contact={bool(record['right_contact'])}",
            f"left_force_n={float(record['left_force_n']):.3f}",
            f"right_force_n={float(record['right_force_n']):.3f}",
            f"root_tilt_deg={float(record['root_tilt_deg']):.3f}",
        )
        line_height = 13
        draw.rectangle((5, 5, 242, 10 + line_height * len(lines)), fill=(0, 0, 0, 170))
        for index, line in enumerate(lines):
            draw.text((10, 8 + line_height * index), line, fill=(255, 255, 255, 255), font=self.font)
        return image

    def _save_keyframe(self, name: str, record: dict[str, Any]) -> None:
        front = self._overlay(self._camera_rgb(self.front_camera), record)
        side = self._overlay(self._camera_rgb(self.side_camera), record)
        self._save_images(name, front, side)
        self.keyframe_snapshots[name] = self._snapshot(record)

    def _save_images(self, name: str, front: Image.Image, side: Image.Image) -> None:
        front.save(self.run_root / f"frame_{name}.png")
        side.save(self.run_root / f"frame_{name}_side.png")

    def _update_actor_metrics(
        self, record: dict[str, Any], front: Image.Image, side: Image.Image
    ) -> None:
        left_gap = float(record["left_actual_surface_gap_m"])
        right_gap = float(record["right_actual_surface_gap_m"])
        frame = int(record["frame"])
        mean_gap = 0.5 * (left_gap + right_gap)
        if left_gap < self.minimum_left_gap_m:
            self.minimum_left_gap_m = left_gap
            self.minimum_left_gap_frame = frame
        if right_gap < self.minimum_right_gap_m:
            self.minimum_right_gap_m = right_gap
            self.minimum_right_gap_frame = frame
        if mean_gap < self.minimum_mean_gap_m:
            self.minimum_mean_gap_m = mean_gap
            self.minimum_mean_gap_frame = frame
            self._minimum_front = front.copy()
            self._minimum_side = side.copy()
            self.keyframe_snapshots["minimum_gap"] = self._snapshot(record)
        self.maximum_force_n[0] = max(self.maximum_force_n[0], float(record["left_force_n"]))
        self.maximum_force_n[1] = max(self.maximum_force_n[1], float(record["right_force_n"]))
        self.contact_seen[0] |= bool(record["left_contact"])
        self.contact_seen[1] |= bool(record["right_contact"])
        self.minimum_root_height_m = min(self.minimum_root_height_m, float(record["root_height_m"]))
        self.maximum_root_tilt_deg = max(self.maximum_root_tilt_deg, float(record["root_tilt_deg"]))
        arm = self.env.action_manager.get_term("arm_residual")
        residual = arm.applied_residual[0].detach().abs().cpu()
        raw = arm.raw_actions[0].detach().abs().cpu()
        self.maximum_abs_residual_rad = max(self.maximum_abs_residual_rad, float(residual.max()))
        for index in range(14):
            self.maximum_abs_residual_per_joint_rad[index] = max(
                self.maximum_abs_residual_per_joint_rad[index], float(residual[index])
            )
            self.maximum_abs_normalized_action_per_joint[index] = max(
                self.maximum_abs_normalized_action_per_joint[index], float(raw[index])
            )
        if self.precontact_palm_world_position is not None:
            palm_pos_w, _ = self._palm_world_pose()
            displacement = torch.linalg.vector_norm(
                palm_pos_w - self.precontact_palm_world_position, dim=-1
            )
            for index in range(2):
                self.maximum_actual_palm_displacement_m[index] = max(
                    self.maximum_actual_palm_displacement_m[index], float(displacement[index])
                )
        if self.precontact_palm_box_position is not None:
            palm_pos_o, _ = self._palm_box_pose()
            displacement_o = torch.linalg.vector_norm(
                palm_pos_o - self.precontact_palm_box_position, dim=-1
            )
            normal_closure = palm_pos_o[:, 0] - self.precontact_palm_box_position[:, 0]
            for index in range(2):
                self.maximum_actual_palm_box_displacement_m[index] = max(
                    self.maximum_actual_palm_box_displacement_m[index], float(displacement_o[index])
                )
                self.maximum_actual_palm_normal_closure_m[index] = max(
                    self.maximum_actual_palm_normal_closure_m[index], float(normal_closure[index])
                )

    def _palm_world_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        palms = self.env.scene["hand_frames"]
        return palms.data.target_pos_w[0], palms.data.target_quat_w[0]

    def _palm_box_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        box = self.env.scene["box"]
        palm_pos_w, palm_quat_w = self._palm_world_pose()
        return math_utils.subtract_frame_transforms(
            box.data.root_link_pos_w[0].expand(2, 3),
            box.data.root_link_quat_w[0].expand(2, 4),
            palm_pos_w,
            palm_quat_w,
        )

    def _snapshot(self, record: dict[str, Any]) -> dict[str, Any]:
        robot = self.env.scene["robot"]
        box = self.env.scene["box"]
        arm = self.env.action_manager.get_term("arm_residual")
        palm_pos_w, palm_quat_w = self._palm_world_pose()
        palm_pos_o, palm_quat_o = math_utils.subtract_frame_transforms(
            box.data.root_link_pos_w[0].expand(2, 3),
            box.data.root_link_quat_w[0].expand(2, 4),
            palm_pos_w,
            palm_quat_w,
        )
        joints = robot.data.joint_pos[0, arm.joint_ids].detach().cpu()
        return {
            "frame": int(record["frame"]),
            "left_arm_joint_positions_rad": joints[0::2].tolist(),
            "right_arm_joint_positions_rad": joints[1::2].tolist(),
            "left_palm_world_pose": {
                "position_m": palm_pos_w[0].detach().cpu().tolist(),
                "quaternion_wxyz": palm_quat_w[0].detach().cpu().tolist(),
            },
            "right_palm_world_pose": {
                "position_m": palm_pos_w[1].detach().cpu().tolist(),
                "quaternion_wxyz": palm_quat_w[1].detach().cpu().tolist(),
            },
            "left_palm_box_pose": {
                "position_m": palm_pos_o[0].detach().cpu().tolist(),
                "quaternion_wxyz": palm_quat_o[0].detach().cpu().tolist(),
            },
            "right_palm_box_pose": {
                "position_m": palm_pos_o[1].detach().cpu().tolist(),
                "quaternion_wxyz": palm_quat_o[1].detach().cpu().tolist(),
            },
            "left_gap_m": float(record["left_actual_surface_gap_m"]),
            "right_gap_m": float(record["right_actual_surface_gap_m"]),
            "left_contact": bool(record["left_contact"]),
            "right_contact": bool(record["right_contact"]),
            "left_force_n": float(record["left_force_n"]),
            "right_force_n": float(record["right_force_n"]),
            "applied_residual_rad": arm.applied_residual[0].detach().cpu().tolist(),
            "normalized_action": arm.raw_actions[0].detach().cpu().tolist(),
            "root_height_m": float(record["root_height_m"]),
            "root_tilt_deg": float(record["root_tilt_deg"]),
        }
