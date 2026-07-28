#!/usr/bin/env python3
"""No-training S2-03T arm-residual action-authority audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
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
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
RUN = args.run_root.resolve()
if RUN.exists():
    raise SystemExit(f"RUN_ROOT_ALREADY_EXISTS:{RUN}")
RUN.mkdir(parents=True)
reference_path = args.reference.resolve()
if sha256_file(reference_path) != args.reference_sha256:
    raise SystemExit("PRECONTACT_REFERENCE_SHA_MISMATCH")
write_json(RUN / "runner_status.json", {"status": "STARTING", "phase": "APP_LAUNCH"})
simulation_app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_actions import ARM_JOINT_NAMES  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_bootstrap import derive_precontact_reference  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env import S203TContactEnv  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_env_cfg import S203TContactEnvCfg  # noqa: E402
from g1_access_push.sim.stage2.s2_03t_mdp import PALM_SUPPORT_OFFSET_M, REAR_FACE_X_M, runtime_state  # noqa: E402

def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return bool(torch.isfinite(value).all())

def object_frame_gaps(env, box_pos: torch.Tensor, box_quat: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    palms = env.scene["hand_frames"]
    palm_w, palm_q = math_utils.combine_frame_transforms(
        robot.data.root_link_pos_w[:, None, :].expand(-1, 2, -1).reshape(-1, 3),
        robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
        palms.data.target_pos_source.reshape(-1, 3),
        palms.data.target_quat_source.reshape(-1, 4),
    )
    palm_o, _ = math_utils.subtract_frame_transforms(
        box_pos[:, None, :].expand(-1, 2, -1).reshape(-1, 3),
        box_quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
        palm_w,
        palm_q,
    )
    return (REAR_FACE_X_M - palm_o.reshape(env.num_envs, 2, 3)[..., 0] - PALM_SUPPORT_OFFSET_M)

def make_sweep_actions(jacobian: torch.Tensor) -> list[torch.Tensor]:
    actions = [torch.zeros(14, device=jacobian.device)]
    for index in range(14):
        for sign in (-1.0, 1.0):
            value = torch.zeros(14, device=jacobian.device)
            value[index] = sign
            actions.append(value)
    generator = torch.Generator(device="cpu").manual_seed(20260728)
    for _ in range(32):
        value = torch.randint(-1, 2, (14,), generator=generator, dtype=torch.int64).to(jacobian.device, torch.float32)
        if bool((value == 0).all()):
            value[0] = 1.0
        actions.append(value)
    gradient = torch.zeros(14, device=jacobian.device)
    for side, action_indices in enumerate((range(0, 14, 2), range(1, 14, 2))):
        row = jacobian[side, 0]
        for local, index in enumerate(action_indices):
            gradient[index] = torch.sign(row[local]) if float(row[local]) != 0.0 else 1.0
    actions.extend((gradient, -gradient))
    return actions

env = None
try:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg = S203TContactEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = S203TContactEnv(cfg=cfg)
    replay_reference, replay_audit = derive_precontact_reference(env)
    env.install_precontact_reference(reference_path)
    installed_diffs = env.installed_reference_tensor_diffs()
    installed_max_diff = max(installed_diffs.values())
    if installed_max_diff > 1.0e-6:
        raise RuntimeError(f"INSTALLED_REFERENCE_MISMATCH:{installed_max_diff}")
    robot = env.scene["robot"]
    box = env.scene["box"]
    arm = env.action_manager.get_term("arm_residual")
    lower = env.action_manager.get_term("frozen_lower_body")
    jacobian = torch.stack((arm._compute_frame_jacobian(0)[0], arm._compute_frame_jacobian(1)[0]), dim=0)
    if tuple(jacobian.shape) != (2, 6, 7) or not finite(jacobian):
        raise RuntimeError("JACOBIAN_INVALID")
    q_reference = robot.data.joint_pos[0, arm.joint_ids].detach().clone()
    limits = robot.data.joint_pos_limits[0, arm.joint_ids].detach().clone()
    original_box_pos = box.data.root_link_pos_w.clone()
    original_box_quat = box.data.root_link_quat_w.clone()
    base_gaps = object_frame_gaps(env, original_box_pos, original_box_quat)[0].detach().clone()
    joint_margin_reference = torch.minimum(q_reference - limits[:, 0], limits[:, 1] - q_reference)
    if not finite(q_reference) or not finite(base_gaps) or not finite(joint_margin_reference):
        raise RuntimeError("REFERENCE_NONFINITE")

    sweep_records = []
    for sweep_index, action in enumerate(make_sweep_actions(jacobian)):
        residual = action * 0.05
        predicted_gaps = base_gaps.clone()
        q_hyp = q_reference + residual
        for side, action_indices in enumerate((range(0, 14, 2), range(1, 14, 2))):
            predicted_gaps[side] = base_gaps[side] - torch.matmul(jacobian[side, 0], residual[list(action_indices)])
        margin = torch.minimum(q_hyp - limits[:, 0], limits[:, 1] - q_hyp)
        sweep_records.append({
            "index": sweep_index,
            "normalized_action": action.detach().cpu().tolist(),
            "residual_rad": residual.detach().cpu().tolist(),
            "predicted_surface_gap_m": predicted_gaps.detach().cpu().tolist(),
            "minimum_joint_margin_rad": float(margin.min()),
            "joint_margin_safe": bool((margin >= 0.10).all()),
            "finite": finite(predicted_gaps) and finite(margin),
        })
    safe_records = [record for record in sweep_records if record["finite"] and record["joint_margin_safe"]]
    if not safe_records:
        raise RuntimeError("NO_FINITE_SAFE_SWEEP_ACTION")
    best = min(safe_records, key=lambda record: sum(record["predicted_surface_gap_m"]) / 2.0)
    gradient_action = torch.tensor(best["normalized_action"], device=env.device, dtype=torch.float32)

    write_json(RUN / "reference_audit.json", {
        "reference_path": str(reference_path),
        "reference_sha256": args.reference_sha256,
        "installed_reference_tensor_diffs": installed_diffs,
        "installed_reference_max_abs_diff": installed_max_diff,
        "bootstrap_replay_audit": replay_audit,
        "joint_names": list(ARM_JOINT_NAMES),
        "joint_order_matches_authoritative": tuple(arm.joint_names) == ARM_JOINT_NAMES,
        "residual_scale_rad": 0.05,
        "maximum_residual_change_rad_per_control_step": 0.005,
        "offset": "frozen arm IK reference",
        "lower_body_checkpoint_sha256": lower.checkpoint_sha256,
        "lower_body_command": [0.0, 0.0, 0.0, 0.7],
    })
    write_json(RUN / "jacobian_audit.json", {
        "jacobian_shape_left_right_6x7": list(jacobian.shape),
        "left_jacobian": jacobian[0].detach().cpu().tolist(),
        "right_jacobian": jacobian[1].detach().cpu().tolist(),
        "base_surface_gap_m": base_gaps.detach().cpu().tolist(),
        "reference_joint_margin_minimum_rad": float(joint_margin_reference.min()),
        "status": "PASS",
    })
    write_json(RUN / "finite_safe_action_sweep.json", {
        "candidate_count": len(sweep_records),
        "safe_candidate_count": len(safe_records),
        "residual_bound_rad": 0.05,
        "mapping": "normalized_action * 0.05 rad, rate limited to 0.005 rad/control step",
        "best_candidate": best,
        "minimum_predicted_left_gap_m": min(record["predicted_surface_gap_m"][0] for record in safe_records),
        "minimum_predicted_right_gap_m": min(record["predicted_surface_gap_m"][1] for record in safe_records),
        "minimum_predicted_bilateral_mean_gap_m": min(sum(record["predicted_surface_gap_m"]) / 2.0 for record in safe_records),
        "records": sweep_records,
    })

    far_box = box.data.root_state_w.clone()
    far_box[:, 0] += 1.0
    far_box[:, 7:] = 0.0
    box.write_root_state_to_sim(far_box)
    env.sim.step()
    env.scene.update(cfg.sim.dt)
    runtime_state(env).reset(torch.tensor([0], device=env.device, dtype=torch.long))
    replay_trace = []
    actions_tensor = gradient_action.unsqueeze(0)
    for step in range(20):
        observation, reward, terminated, truncated, _ = env.step(actions_tensor)
        gaps = object_frame_gaps(env, original_box_pos, original_box_quat)
        metric = runtime_state(env).ensure()
        replay_trace.append({
            "step": step + 1,
            "gaps_against_original_box_m": gaps[0].detach().cpu().tolist(),
            "action_finite": finite(actions_tensor),
            "observation_finite": finite(observation),
            "reward_finite": finite(reward),
            "terminated": bool(terminated[0]),
            "truncated": bool(truncated[0]),
            "root_height_m": float(metric["root_height"][0]),
            "root_tilt_deg": float(metric["root_tilt_deg"][0]),
            "joint_margin_minimum_rad": float(metric["arm_joint_margin"][0]),
            "torque_ratio_maximum": float(metric["arm_torque_ratio"][0]),
            "forbidden_collision": bool(metric["failure_forbidden_non_palm_box_collision"][0]),
            "contact_force_maximum_n": float(metric["forces"][0].max()),
        })
        if (step + 1) % 5 == 0:
            print(f"PHASE=SAFE_COUNTERFACTUAL_REPLAY step={step + 1}", flush=True)
    with (RUN / "safe_counterfactual_trace.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
        for item in replay_trace:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")

    max_tilt = max(item["root_tilt_deg"] for item in replay_trace)
    min_height = min(item["root_height_m"] for item in replay_trace)
    min_margin = min(item["joint_margin_minimum_rad"] for item in replay_trace)
    max_torque = max(item["torque_ratio_maximum"] for item in replay_trace)
    any_collision = any(item["forbidden_collision"] for item in replay_trace)
    any_contact = any(item["contact_force_maximum_n"] >= 1.0 for item in replay_trace)
    predicted_contact = best["predicted_surface_gap_m"][0] <= 0.001 and best["predicted_surface_gap_m"][1] <= 0.001
    authority_status = "SUFFICIENT_FOR_CONTACT" if predicted_contact and not any_collision else "INSUFFICIENT_FOR_CONTACT"
    primary_reason = "FINITE_SAFE_JACOBIAN_SWEEP_REACHES_BOTH_SURFACES" if predicted_contact else "ARM_RESIDUAL_AUTHORITY_INSUFFICIENT"
    write_json(RUN / "result.json", {
        "schema_version": 1,
        "stage": "S2-03T",
        "audit": "ACTION_AUTHORITY_AUDIT",
        "status": "PASS",
        "primary_reason": primary_reason,
        "authority_status": authority_status,
        "contact_not_attempted": True,
        "box_push_commanded": False,
        "training_started": False,
        "checkpoint_loaded": False,
        "same_frozen_ik_reference": True,
        "same_joint_order": tuple(arm.joint_names) == ARM_JOINT_NAMES,
        "residual_bound_rad": 0.05,
        "minimum_predicted_surface_gap_m": best["predicted_surface_gap_m"],
        "minimum_actual_counterfactual_gap_m": [min(item["gaps_against_original_box_m"][side] for item in replay_trace) for side in range(2)],
        "safe_replay_metrics": {
            "observed_steps": len(replay_trace),
            "minimum_root_height_m": min_height,
            "maximum_root_tilt_deg": max_tilt,
            "minimum_joint_margin_rad": min_margin,
            "maximum_torque_ratio": max_torque,
            "forbidden_collision": any_collision,
            "contact_force_observed": any_contact,
            "finite": all(item["action_finite"] and item["observation_finite"] and item["reward_finite"] for item in replay_trace),
        },
        "diagnostic_valid": True,
    })
    write_json(RUN / "runner_status.json", {"status": "COMPLETE", "phase": "RESULT_READY"})
except BaseException as exc:
    traceback.print_exc()
    write_json(RUN / "runner_status.json", {"status": "INVALID", "phase": "EXCEPTION", "primary_reason": "IMPLEMENTATION_EXCEPTION", "error": repr(exc)})
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
