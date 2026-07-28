#!/usr/bin/env python3
"""Five-gap deterministic reachability gate for the redesigned S2-03T action."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from g1_access_push.stage2.s2_03t_redesign_contract import (
    validate_campaign_execution_manifest,
)

EXPECTED_GAPS_M = (0.0, 0.005, 0.010, 0.030, 0.060)
EXPECTED_STEPS = 1000
INITIAL_GAP_PLACEMENT_TOLERANCE_M = 0.001
PUSH_REASONS = {
    "box_linear_speed",
    "box_angular_speed",
    "box_translation",
    "box_yaw_change",
}
HARD_SAFETY_REASONS = {
    "nonfinite",
    "forbidden_non_palm_box_collision",
    "force_peak",
    "palm_impulse",
    "combined_impulse",
    "force_rate",
    "base_excursion",
    "root_height",
    "root_tilt",
    "arm_joint_margin",
    "arm_torque",
}
REQUIRED_EXECUTION_SOURCES = (
    "scripts/stage2_isaac/run_s2_03t_reachability_smoke.py",
    "scripts/stage2_isaac/s2_03t_visual_recorder.py",
    "src/g1_access_push/sim/stage2/s2_03t_actions.py",
    "src/g1_access_push/sim/stage2/s2_03t_env.py",
    "src/g1_access_push/sim/stage2/s2_03t_env_cfg.py",
    "src/g1_access_push/sim/stage2/s2_03t_mdp.py",
    "src/g1_access_push/stage2/s2_03t_redesign_contract.py",
)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


parser = argparse.ArgumentParser()
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--reference-sha256", required=True)
parser.add_argument("--campaign-manifest", type=Path, required=True)
parser.add_argument("--initial-gaps-m", type=float, nargs="+", required=True)
parser.add_argument("--record-diagnostic-on-failure", action="store_true")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
requested_gaps = tuple(round(value, 6) for value in args.initial_gaps_m)
if requested_gaps != tuple(round(value, 6) for value in EXPECTED_GAPS_M):
    parser.error(f"reachability gaps must be exactly {EXPECTED_GAPS_M}")
if args.record_diagnostic_on_failure:
    # Cameras are configured up front but read only for a post-result diagnostic
    # replay if a scientific case fails.  The five qualification cases remain
    # camera-independent and authoritative.
    args.enable_cameras = True

RUN = args.run_root.resolve()
RUN.mkdir(parents=True, exist_ok=False)
RESULT = RUN / "reachability_result.json"
STATUS = RUN / "reachability_status.json"
reference_path = args.reference.resolve()
if not reference_path.is_file() or sha256_file(reference_path) != args.reference_sha256:
    raise SystemExit("PRECONTACT_REFERENCE_SHA_MISMATCH")
manifest = validate_campaign_execution_manifest(
    args.campaign_manifest.resolve(),
    stage_run_root=RUN,
    reference_path=reference_path,
    reference_sha256=args.reference_sha256,
    required_source_relatives=REQUIRED_EXECUTION_SOURCES,
)
atomic_write_json(
    STATUS, {"status": "STARTING", "phase": "APP_LAUNCH", "timestamp_utc": utc_timestamp()}
)
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from s2_03t_visual_recorder import _RawVideoWriter  # noqa: E402

from g1_access_push.sim.stage2.s2_03t_bootstrap import derive_precontact_reference  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env import S203TContactEnv  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env_cfg import S203TContactEnvCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_mdp import runtime_state  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_visual_cfg import S203TVisualEvidenceEnvCfg  # noqa: E402


class TerminalCaptureS203TContactEnv(S203TContactEnv):
    """Expose the exact terminal control frame before automatic reset."""

    def __init__(self, *env_args, **env_kwargs) -> None:
        self.terminal_callback = None
        super().__init__(*env_args, **env_kwargs)

    def _reset_idx(self, env_ids) -> None:
        if (
            self.terminal_callback is not None
            and hasattr(self, "episode_length_buf")
            and not self._s2_03t_bootstrap_mode
        ):
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            completed = ids[self.episode_length_buf[ids] > 0]
            if bool((completed == 0).any()):
                self.terminal_callback()
        super()._reset_idx(env_ids)


def finite_metric(metric: dict[str, torch.Tensor]) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for value in metric.values()
        if isinstance(value, torch.Tensor)
    )


def case_result(
    initial_gap_m: float, snapshot: dict[str, Any], trace: list[dict[str, Any]]
) -> dict[str, Any]:
    reasons = set(snapshot["termination_reasons"])
    hard = sorted(reasons & HARD_SAFETY_REASONS)
    pushing = sorted(reasons & PUSH_REASONS)
    contact = bool(snapshot["bilateral_contact_seen"])
    verify = int(snapshot["verify_steps"]) >= 20
    hold = int(snapshot["attached_hold_steps"]) >= 100
    trace_finite = all(bool(item["finite"]) for item in trace)
    passed = bool(
        snapshot["success"]
        and contact
        and verify
        and hold
        and not hard
        and not pushing
        and snapshot["finite"]
        and trace_finite
    )
    return {
        "initial_gap_m": initial_gap_m,
        "status": "PASS" if passed else "FAIL",
        "primary_reason": "SAFE_BILATERAL_VERIFY_HOLD"
        if passed
        else (
            hard[0].upper()
            if hard
            else pushing[0].upper()
            if pushing
            else "CONTACT_VERIFY_HOLD_NOT_COMPLETED"
        ),
        "episode_steps": int(snapshot["episode_steps"]),
        "time_out": bool(snapshot["time_out"]),
        "bilateral_contact": contact,
        "verify": verify,
        "safe_hold": hold,
        "verify_steps": int(snapshot["verify_steps"]),
        "attached_hold_steps": int(snapshot["attached_hold_steps"]),
        "contact_retention_steps": int(snapshot.get("contact_retention_steps", 0)),
        "minimum_surface_gap_m": snapshot["minimum_surface_gap_m"],
        "maximum_force_n": snapshot["maximum_force_n"],
        "impulse_ns": snapshot["impulse_ns"],
        "maximum_box_translation_m": snapshot["maximum_box_translation_m"],
        "maximum_box_yaw_change_rad": snapshot["maximum_box_yaw_change_rad"],
        "minimum_root_height_m": snapshot["minimum_root_height_m"],
        "maximum_root_tilt_deg": snapshot["maximum_root_tilt_deg"],
        "minimum_arm_joint_margin_rad": snapshot["minimum_arm_joint_margin_rad"],
        "maximum_arm_torque_ratio": snapshot["maximum_arm_torque_ratio"],
        "hard_safety_reasons": hard,
        "pushing_reasons": pushing,
        "termination_reasons": snapshot["termination_reasons"],
        "finite": bool(snapshot["finite"]),
        "trace_finite": trace_finite,
        "observed_control_steps": len(trace),
        "no_auto_reset_accumulation": True,
        "actor_loaded": False,
        "ppo_started": False,
        "box_push_commanded": False,
    }


def write_trace(path: Path, trace: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in trace:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")


def trace_record(step: int, metric: dict[str, torch.Tensor], arm: Any) -> dict[str, Any]:
    return {
        "step": step,
        "finite": finite_metric(metric),
        "left_gap_m": float(metric["gaps"][0, 0]),
        "right_gap_m": float(metric["gaps"][0, 1]),
        "left_contact": bool(metric["contacts"][0, 0]),
        "right_contact": bool(metric["contacts"][0, 1]),
        "left_force_n": float(metric["forces"][0, 0]),
        "right_force_n": float(metric["forces"][0, 1]),
        "verify_count": int(runtime_state(env).verify_count[0]),
        "hold_count": int(runtime_state(env).hold_count[0]),
        "nominal_displacement_m": [float(value) for value in arm.nominal_displacement_m[0]],
        "learned_correction_m": [float(value) for value in arm.correction_m[0]],
        "raw_action": [float(value) for value in arm.raw_actions[0]],
        "box_translation_m": float(metric["box_translation"][0]),
        "root_tilt_deg": float(metric["root_tilt_deg"][0]),
        "capture": "PRE_AUTO_RESET_PHYSICS_FRAME",
    }


def camera_rgb(camera: Any) -> np.ndarray:
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    return rgb.astype(np.uint8, copy=False)


def overlay(
    rgb: np.ndarray, metric: dict[str, torch.Tensor], frame: int, gap_m: float
) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")
    lines = (
        "DIAGNOSTIC REPLAY - NOT QUALIFICATION",
        f"requested_initial_gap_m={gap_m:.3f}",
        f"frame={frame}",
        f"left_gap_m={float(metric['gaps'][0, 0]):.5f}",
        f"right_gap_m={float(metric['gaps'][0, 1]):.5f}",
        f"left_force_n={float(metric['forces'][0, 0]):.3f}",
        f"right_force_n={float(metric['forces'][0, 1]):.3f}",
    )
    font = ImageFont.load_default()
    draw.rectangle((5, 5, 310, 12 + 14 * len(lines)), fill=(0, 0, 0, 180))
    for index, line in enumerate(lines):
        draw.text((10, 8 + 14 * index), line, fill=(255, 255, 255, 255), font=font)
    return np.asarray(image, dtype=np.uint8)


env = None
wrapped = None
scientific_rc = 2
try:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg = S203TVisualEvidenceEnvCfg() if args.record_diagnostic_on_failure else S203TContactEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = TerminalCaptureS203TContactEnv(
        cfg=cfg,
        render_mode="rgb_array" if args.record_diagnostic_on_failure else None,
    )
    if args.record_diagnostic_on_failure:
        front_camera = env.scene["audit_camera"]
        side_camera = env.scene["side_camera"]
        front_camera.set_world_poses_from_view(
            torch.tensor([[-1.6, -2.2, 1.40]], device=env.device),
            torch.tensor([[0.45, 0.0, 0.70]], device=env.device),
        )
        side_camera.set_world_poses_from_view(
            torch.tensor([[0.55, -2.35, 0.92]], device=env.device),
            torch.tensor([[0.55, 0.0, 0.68]], device=env.device),
        )
    else:
        front_camera = side_camera = None

    replay_reference, replay_audit = derive_precontact_reference(env)
    stored_reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    tensor_names = (
        "robot_root_state_relative",
        "robot_joint_position",
        "robot_joint_velocity",
        "box_root_state_relative",
        "arm_ik_target",
        "lower_hidden_state",
        "lower_cell_state",
        "previous_lower_policy_action",
    )
    tensor_diffs = {
        name: float(torch.max(torch.abs(replay_reference[name] - stored_reference[name])))
        for name in tensor_names
    }
    replay_max_diff = max(tensor_diffs.values())
    if not math.isfinite(replay_max_diff):
        raise RuntimeError("PRECONTACT_REFERENCE_REPLAY_NONFINITE")
    env.configure_contact_curriculum(enabled=False, maximum_level=0, fixed_gap_m=0.06)
    env.install_precontact_reference(stored_reference)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    arm = env.action_manager.get_term("arm_residual")
    if arm.action_dim != 2:
        raise RuntimeError(f"HYBRID_ACTION_DIM_MISMATCH:{arm.action_dim}")

    cases: list[dict[str, Any]] = []
    traces: dict[float, list[dict[str, Any]]] = {}
    for case_index, gap_m in enumerate(EXPECTED_GAPS_M):
        env.configure_contact_curriculum(enabled=False, maximum_level=0, fixed_gap_m=gap_m)
        env._s2_03t_terminal_snapshots.clear()
        env.terminal_callback = None
        env.reset(seed=42)
        observation, _ = wrapped.get_observations()
        initial_metric = runtime_state(env).ensure()
        measured_initial_gap = [
            float(initial_metric["gaps"][0, 0]),
            float(initial_metric["gaps"][0, 1]),
        ]
        configured_initial_gap = float(env.sampled_initial_gap_m[0])
        if tuple(observation.shape) != (1, 135) or not bool(torch.isfinite(observation).all()):
            raise RuntimeError(f"REACHABILITY_OBSERVATION_INVALID:{tuple(observation.shape)}")
        trace: list[dict[str, Any]] = []
        terminal_trace: list[dict[str, Any]] = []
        current_step = [0]

        def capture_case_terminal(
            terminal_trace: list[dict[str, Any]] = terminal_trace,
            current_step: list[int] = current_step,
        ) -> None:
            terminal_trace.append(trace_record(current_step[0], runtime_state(env).ensure(), arm))

        env.terminal_callback = capture_case_terminal
        terminal_snapshot = None
        for step in range(EXPECTED_STEPS):
            zero_action = torch.zeros((1, 2), device=env.device)
            current_step[0] = step + 1
            observation, reward, done, _ = wrapped.step(zero_action)
            if not bool(torch.isfinite(observation).all() and torch.isfinite(reward).all()):
                raise RuntimeError(f"REACHABILITY_NONFINITE:gap={gap_m}:step={step}")
            if bool(done[0]):
                terminal_snapshot = env.pop_terminal_snapshot(0)
                if terminal_snapshot is None:
                    raise RuntimeError(f"TERMINAL_SNAPSHOT_MISSING:gap={gap_m}")
                if len(terminal_trace) != 1:
                    raise RuntimeError(
                        f"TERMINAL_PRE_RESET_TRACE_MISSING:gap={gap_m}:count={len(terminal_trace)}"
                    )
                trace.append(terminal_trace[0])
                break
            metric = runtime_state(env).ensure()
            if not finite_metric(metric):
                raise RuntimeError(f"REACHABILITY_NONFINITE:gap={gap_m}:step={step}")
            trace.append(trace_record(step + 1, metric, arm))
            if (step + 1) % 25 == 0:
                print(f"PHASE=REACHABILITY gap_m={gap_m:.3f} step={step + 1}", flush=True)
        if terminal_snapshot is None:
            raise RuntimeError(f"REACHABILITY_TERMINAL_EVIDENCE_MISSING:gap={gap_m}")
        case = case_result(gap_m, terminal_snapshot, trace)
        initial_gap_errors = [abs(value - gap_m) for value in measured_initial_gap]
        case["configured_initial_gap_m"] = configured_initial_gap
        case["measured_initial_surface_gap_m"] = measured_initial_gap
        case["initial_gap_abs_error_m"] = initial_gap_errors
        case["initial_gap_max_abs_error_m"] = max(initial_gap_errors)
        case["configured_initial_gap_exact"] = math.isclose(
            configured_initial_gap, gap_m, rel_tol=0.0, abs_tol=1.0e-7
        )
        case["measured_initial_gap_within_placement_tolerance"] = bool(
            max(initial_gap_errors) <= INITIAL_GAP_PLACEMENT_TOLERANCE_M
        )
        case["maximum_abs_learned_correction_m"] = max(
            (max(abs(value) for value in item["learned_correction_m"]) for item in trace),
            default=0.0,
        )
        case["maximum_abs_raw_action"] = max(
            (max(abs(value) for value in item["raw_action"]) for item in trace),
            default=0.0,
        )
        case["trace_path"] = str(RUN / f"gap_{int(round(gap_m * 1000)):03d}mm" / "trace.jsonl")
        write_trace(Path(case["trace_path"]), trace)
        traces[gap_m] = trace
        cases.append(case)
        atomic_write_json(
            STATUS,
            {
                "status": "RUNNING",
                "phase": "QUALIFICATION",
                "completed_cases": case_index + 1,
                "current_gap_m": gap_m,
                "timestamp_utc": utc_timestamp(),
            },
        )

    diagnostic_valid = bool(
        len(cases) == len(EXPECTED_GAPS_M)
        and all(case["finite"] for case in cases)
        and all(case["trace_finite"] for case in cases)
        and all(case["observed_control_steps"] == case["episode_steps"] for case in cases)
        and all(case["configured_initial_gap_exact"] for case in cases)
        and all(case["measured_initial_gap_within_placement_tolerance"] for case in cases)
        and all(case["maximum_abs_learned_correction_m"] == 0.0 for case in cases)
        and all(case["maximum_abs_raw_action"] == 0.0 for case in cases)
    )
    passed = diagnostic_valid and all(case["status"] == "PASS" for case in cases)
    if not diagnostic_valid:
        status, reason = "INVALID", "REACHABILITY_EVIDENCE_INCOMPLETE"
    elif passed:
        status, reason = "PASS", "ALL_FIVE_GAPS_SAFE_REACHABLE"
    else:
        status = "FAIL"
        required_failures = [
            case
            for case in cases
            if case["initial_gap_m"] in (0.03, 0.06) and case["status"] != "PASS"
        ]
        reason = (
            "REQUIRED_30MM_OR_60MM_GAP_NOT_REACHABLE"
            if required_failures
            else "ONE_OR_MORE_GAPS_NOT_REACHABLE"
        )
    result = {
        "schema_version": 1,
        "stage": "S2_03T_NEW_ACTION_REACHABILITY_SMOKE",
        "status": status,
        "primary_reason": reason,
        "diagnostic_valid": diagnostic_valid,
        "authoritative_complete_before_teardown": True,
        "cases": cases,
        "case_count": len(cases),
        "reference": {"path": str(reference_path), "sha256": args.reference_sha256},
        "reference_replay_max_abs_diff_diagnostic_only": replay_max_diff,
        "reference_replay_tensor_diffs": tensor_diffs,
        "reference_replay_audit": replay_audit,
        "action": {
            "name": "HYBRID_NOMINAL_NORMAL_APPROACH_PLUS_LEARNED_BILATERAL_NORMAL_CORRECTION",
            "public_dim": 2,
            "actor_loaded": False,
            "learned_action": [0.0, 0.0],
        },
        "execution": {
            "num_envs": 1,
            "episode_count": 5,
            "ppo_started": False,
            "checkpoint_loaded": False,
            "auto_reset_accumulation": False,
            "box_push_commanded": False,
            "planner_started": False,
            "qualification_camera_data_read": False,
            "terminal_trace_capture": "PRE_AUTO_RESET_PHYSICS_FRAME",
            "post_done_steps_executed": 0,
            "initial_gap_placement_tolerance_m": INITIAL_GAP_PLACEMENT_TOLERANCE_M,
        },
        "diagnostic_replay": {
            "requested": bool(args.record_diagnostic_on_failure),
            "status": "NOT_REQUIRED" if status == "PASS" else "PENDING",
            "scientific_gate": False,
        },
        "completed_at_utc": utc_timestamp(),
    }
    atomic_write_json(RESULT, result)
    scientific_rc = 0 if status in {"PASS", "FAIL"} else 2

    if status == "FAIL" and args.record_diagnostic_on_failure:
        failed = [case for case in cases if case["status"] == "FAIL"]
        selected = max(
            failed,
            key=lambda item: (
                bool(item["hard_safety_reasons"] or item["pushing_reasons"]),
                item["initial_gap_m"] in (0.03, 0.06),
                item["initial_gap_m"],
            ),
        )
        selected_gap = float(selected["initial_gap_m"])
        diagnostic_root = RUN / "diagnostic_failure_replay"
        diagnostic_root.mkdir()
        front_path = diagnostic_root / "failure_front.mp4"
        side_path = diagnostic_root / "failure_side.mp4"
        front_writer = _RawVideoWriter(front_path, 640, 480, 25)
        side_writer = _RawVideoWriter(side_path, 640, 480, 25)
        diagnostic_status = "PASS"
        diagnostic_reason = "MOST_DIAGNOSTIC_FAILED_GAP_REPLAY_RECORDED"
        try:
            env.terminal_callback = None
            env.configure_contact_curriculum(
                enabled=False, maximum_level=0, fixed_gap_m=selected_gap
            )
            env._s2_03t_terminal_snapshots.clear()
            env.reset(seed=42)
            observation, _ = wrapped.get_observations()
            replay_steps = 0
            diagnostic_step = [0]
            terminal_frames: dict[str, np.ndarray] = {}

            def capture_diagnostic_terminal() -> None:
                terminal_metric = runtime_state(env).ensure()
                terminal_frames["front"] = overlay(
                    camera_rgb(front_camera), terminal_metric, diagnostic_step[0], selected_gap
                ).copy()
                terminal_frames["side"] = overlay(
                    camera_rgb(side_camera), terminal_metric, diagnostic_step[0], selected_gap
                ).copy()

            env.terminal_callback = capture_diagnostic_terminal
            for step in range(EXPECTED_STEPS):
                diagnostic_step[0] = step + 1
                observation, _, done, _ = wrapped.step(torch.zeros((1, 2), device=env.device))
                replay_steps = step + 1
                if bool(done[0]):
                    if set(terminal_frames) != {"front", "side"}:
                        raise RuntimeError("DIAGNOSTIC_TERMINAL_PRE_RESET_FRAME_MISSING")
                    front_writer.write(terminal_frames["front"])
                    side_writer.write(terminal_frames["side"])
                    Image.fromarray(terminal_frames["front"], mode="RGB").save(
                        diagnostic_root / "failure_terminal_front.png"
                    )
                    Image.fromarray(terminal_frames["side"], mode="RGB").save(
                        diagnostic_root / "failure_terminal_side.png"
                    )
                    break
                metric = runtime_state(env).ensure()
                if step % 2 == 0:
                    front_writer.write(
                        overlay(camera_rgb(front_camera), metric, step + 1, selected_gap)
                    )
                    side_writer.write(
                        overlay(camera_rgb(side_camera), metric, step + 1, selected_gap)
                    )
            front_writer.close()
            side_writer.close()
            terminal_front = diagnostic_root / "failure_terminal_front.png"
            terminal_side = diagnostic_root / "failure_terminal_side.png"
            diagnostic = {
                "status": diagnostic_status,
                "primary_reason": diagnostic_reason,
                "scientific_gate": False,
                "selected_initial_gap_m": selected_gap,
                "replay_steps": replay_steps,
                "terminal_capture": "PRE_AUTO_RESET_PHYSICS_FRAME",
                "terminal_frame_included_in_videos": True,
                "front_video": {
                    "path": str(front_path),
                    "size_bytes": front_path.stat().st_size,
                    "sha256": sha256_file(front_path),
                },
                "side_video": {
                    "path": str(side_path),
                    "size_bytes": side_path.stat().st_size,
                    "sha256": sha256_file(side_path),
                },
                "terminal_front_image": {
                    "path": str(terminal_front),
                    "size_bytes": terminal_front.stat().st_size,
                    "sha256": sha256_file(terminal_front),
                },
                "terminal_side_image": {
                    "path": str(terminal_side),
                    "size_bytes": terminal_side.stat().st_size,
                    "sha256": sha256_file(terminal_side),
                },
            }
        except BaseException as diagnostic_exc:
            diagnostic_status = "INVALID"
            diagnostic_reason = f"DIAGNOSTIC_REPLAY_EXCEPTION:{diagnostic_exc!r}"
            for writer in (front_writer, side_writer):
                try:
                    writer.close()
                except BaseException:
                    pass
            diagnostic = {
                "status": diagnostic_status,
                "primary_reason": diagnostic_reason,
                "scientific_gate": False,
                "selected_initial_gap_m": selected_gap,
            }
        result["diagnostic_replay"] = diagnostic
        atomic_write_json(RESULT, result)

    atomic_write_json(
        STATUS,
        {
            "status": "COMPLETE" if status in {"PASS", "FAIL"} else "INVALID",
            "result": status,
            "timestamp_utc": utc_timestamp(),
        },
    )
except BaseException as exc:
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    if not RESULT.is_file():
        atomic_write_json(
            RESULT,
            {
                "schema_version": 1,
                "stage": "S2_03T_NEW_ACTION_REACHABILITY_SMOKE",
                "status": "INVALID",
                "primary_reason": "REACHABILITY_IMPLEMENTATION_EXCEPTION",
                "diagnostic_valid": False,
                "authoritative_complete_before_teardown": True,
                "error": repr(exc),
                "completed_at_utc": utc_timestamp(),
            },
        )
    atomic_write_json(
        STATUS, {"status": "INVALID", "error": repr(exc), "timestamp_utc": utc_timestamp()}
    )
    scientific_rc = 2
finally:
    if env is not None:
        env.close()
    simulation_app.close()

raise SystemExit(scientific_rc)
