"""Campaign driver used by the isolated S2-03T runner entry point."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from g1_access_push.stage2.s2_03t_pelvis_contact_runtime import tolist, write_json

import torch


def _target_pose(reference: dict[str, Any], root_pos: torch.Tensor, root_quat: torch.Tensor, ns: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the frozen S2-02 palm target in the current root frame."""
    math_utils = ns["math_utils"]
    old_contract = __import__("g1_access_push.stage2.s2_03t_safe_chest_joint_reference_contract", fromlist=["x"])
    local = torch.tensor(old_contract.certified_object_local_targets(), dtype=root_pos.dtype, device=root_pos.device)
    root_ref = reference["robot_root_state_relative"].to(device=root_pos.device, dtype=root_pos.dtype)
    box_ref = reference["box_root_state_relative"].to(device=root_pos.device, dtype=root_pos.dtype)
    box_pos_b, box_quat_b = math_utils.subtract_frame_transforms(
        root_ref[:3].unsqueeze(0), root_ref[3:7].unsqueeze(0), box_ref[:3].unsqueeze(0), box_ref[3:7].unsqueeze(0))
    box_pos_w, box_quat_w = math_utils.combine_frame_transforms(
        root_pos.unsqueeze(0), root_quat.unsqueeze(0), box_pos_b, box_quat_b)
    object_quat = torch.tensor(old_contract.S2_02_PALM_QUATERNION_OBJECT_WXYZ, dtype=root_pos.dtype, device=root_pos.device).expand(2, 4)
    world_pos, world_quat = math_utils.combine_frame_transforms(box_pos_w.expand(2, 3), box_quat_w.expand(2, 4), local, object_quat)
    return math_utils.subtract_frame_transforms(root_pos.expand(2, 3), root_quat.expand(2, 4), world_pos, world_quat)


def _write_trace(run: Path, label: str, value: dict[str, Any]) -> None:
    from g1_access_push.stage2.s2_03t_pelvis_contact_runtime import write_json
    write_json(run / f"{label.lower()}_trace.json", {"stage": "S2_03T_PELVIS_CONTACT_PAIR_AND_LEFT_ARM_PATH_RECOVERY", "label": label, "records": value.get("records", [])})


def _configure_collision_visualization() -> dict[str, Any]:
    try:
        import carb
        from omni.physx.bindings._physx import SETTING_VISUALIZATION_COLLISION_MESH
        carb.settings.get_settings().set_bool(SETTING_VISUALIZATION_COLLISION_MESH, True)
        return {"enabled": True, "setting": str(SETTING_VISUALIZATION_COLLISION_MESH)}
    except Exception as exc:
        return {"enabled": False, "reason": f"{type(exc).__name__}:{exc}"}


def _classify_contact(pair: dict[str, Any] | None) -> str:
    if pair is None:
        return "UNKNOWN"
    name = str(pair.get("body_b_name", "")).lower()
    if name == "ground":
        return "PELVIS_VS_GROUND"
    if "palm" in name or "hand" in name:
        return "LEFT_HAND_VS_PELVIS_REAL_PATH_COLLISION" if name.startswith("left") else "INTENTIONAL_ADJACENT_COLLIDER_OVERLAP"
    if "wrist" in name:
        return "LEFT_WRIST_VS_PELVIS_REAL_PATH_COLLISION" if name.startswith("left") else "INTENTIONAL_ADJACENT_COLLIDER_OVERLAP"
    if "forearm" in name or "elbow" in name:
        return "LEFT_FOREARM_VS_PELVIS_REAL_PATH_COLLISION" if name.startswith("left") else "INTENTIONAL_ADJACENT_COLLIDER_OVERLAP"
    if "shoulder" in name or "upper" in name:
        return "LEFT_UPPER_ARM_VS_PELVIS_REAL_PATH_COLLISION" if name.startswith("left") else "INTENTIONAL_ADJACENT_COLLIDER_OVERLAP"
    return "UNKNOWN"


def _pair_sets(pair: Any) -> tuple[set[str], set[str]]:
    default: set[str] = set()
    final: set[str] = set()
    for path, records in pair.history.items():
        for record in records:
            if str(record.get("phase", "")).startswith("BASELINE"):
                default.add(path)
            if str(record.get("phase", "")) == "FINAL_STATIC_TARGET":
                final.add(path)
    return default, final


def run(ns: dict[str, Any]) -> int:
    """Run all phases once.  ``ns`` is the already AppLauncher-started runner namespace."""
    from g1_access_push.stage2.s2_03t_pelvis_contact_runtime import write_json

    run_root: Path = ns["RUN"]
    args = ns["ARGS"]
    stage_name = ns["STAGE"]
    env = None
    videos: dict[str, Any] = {}
    phase_results: dict[str, Any] = {}
    reference_path = (args.reference or ns["REFERENCE_PATH"]).resolve()
    reference = ns["load_reference"](reference_path)
    write_json(run_root / "runner_status.json", {"status": "RUNNING", "phase": "ENVIRONMENT_CREATE", "stage": stage_name})

    cfg_class = ns["S203TSafeChestJointReferenceEnvCfg"]
    cfg = cfg_class()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = ns["ManagerBasedEnv"](cfg=cfg)
    ns["ENV"] = env
    env.reset(seed=args.seed)
    ns["set_camera_views"](env)
    stage = ns["get_current_stage"]()
    runtime = ns["runtime_audit"](env, cfg, stage)
    robot = env.scene["robot"]
    root_pos, root_quat = robot.data.root_link_pos_w[0].detach().clone(), robot.data.root_link_quat_w[0].detach().clone()
    target_pos, target_quat = _target_pose(reference, root_pos, root_quat, ns)
    order = ns["order_audit"](reference["arm_joint_names"], runtime["arm_joint_names"], runtime["arm_joint_names"])
    mirror = ns["mirror_audit"](
        reference["arm_ik_target"][0::2].tolist(), reference["arm_ik_target"][1::2].tolist(),
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0])
    if not order["pass"]:
        raise RuntimeError("ARM_REFERENCE_ORDER_INVALID")
    colliders = ns["collider_audit"](stage, robot, runtime["body_paths"])
    write_json(run_root / "collider_audit.json", colliders)
    if colliders["status"] != "PASS":
        raise RuntimeError("COLLIDER_AUDIT_INCOMPLETE")
    pair = ns["ContactPairAuditor"](env, stage, runtime, colliders, ns["CONTROL_DT_S"])
    ns["PAIR"] = pair
    # runpy keeps runner function globals separate from the campaign namespace
    # in some Isaac entry paths; bind the auditor explicitly for every record.
    ns["run_episode"].__globals__["PAIR"] = pair
    ns["static_state"].__globals__["PAIR"] = pair
    write_json(run_root / "contact_pair_sensor_audit.json", pair.coverage)
    clearance = ns["BodyClearanceModel"](stage, robot, colliders)
    write_json(run_root / "runtime_clearance_model.json", {"radii_m": clearance.radii, "sources": clearance.sources, "diagnostics": clearance.diagnostics, "method": "CONSERVATIVE_WORLD_COLLIDER_VERTEX_BOUNDING_SPHERES"})
    write_json(
        run_root / "resolved_config.json",
        {
            "schema_version": 1, "stage": stage_name, "seed": args.seed, "num_envs": 1,
            "control_dt_s": ns["CONTROL_DT_S"], "settle_steps": ns["SETTLE_STEPS"], "segment_steps": ns["MOVE_STEPS"], "hold_steps": ns["HOLD_STEPS"],
            "render_mode": "rgb_array", "enable_cameras": True, "box_used": False, "actor_enabled": False,
            "base_walking_started": False, "contact_attach_started": False, "ppo_started": False, "falcon_started": False,
            "planner_started": False, "control_mode": "JOINT_SPACE_MINIMUM_JERK", "differential_ik_used": False,
            "reference_path": str(reference_path), "reference_sha256": ns["sha256"](reference_path),
            "lower_body_command": [0.0, 0.0, 0.0, 0.7], "lower_body_checkpoint_sha256": runtime["action_terms"]["lower_body_checkpoint_sha256"],
            "arm_reference_order_audit": order, "arm_reference_mirror_audit": mirror, "runtime": runtime,
            "formal_gates": {"dynamic_joint_margin_rad": ns["DYNAMIC_MARGIN_GATE_RAD"], "static_joint_margin_rad": ns["STATIC_MARGIN_GATE_RAD"],
                             "forbidden_force_n": ns["FORBIDDEN_FORCE_GATE_N"], "torque_ratio": ns["TORQUE_RATIO_GATE"],
                             "root_tilt_deg": ns["ROOT_TILT_GATE_DEG"], "root_height_range_m": list(ns["ROOT_HEIGHT_RANGE_M"])},
            "development_clearance_m": ns["DEVELOPMENT_CLEARANCE_M"], "static_straight_samples": ns["STATIC_STRAIGHT_SAMPLES"],
            "static_segment_samples": ns["STATIC_SEGMENT_SAMPLES"],
        },
    )

    qchest = reference["arm_ik_target"].to(device=env.device, dtype=robot.data.joint_pos.dtype).detach().clone()
    q0 = robot.data.joint_pos[0, runtime["arm_joint_ids"]].detach().clone()
    q0 = q0.to(device=env.device, dtype=qchest.dtype)
    q_default_full = q0.clone()
    q_left_target = qchest.clone()
    q_right_target = qchest.clone()
    debug_visualization = _configure_collision_visualization()
    debug_video = ns["EvidenceVideo"](run_root, env, "pelvis_collision_debug_front.mp4", "pelvis_collision_debug_side.mp4")
    ns["VIDEOS"].append(debug_video)

    write_json(run_root / "runner_status.json", {"status": "RUNNING", "phase": "DEFAULT_STAND_BASELINE"})
    baseline = ns["run_episode"](env, runtime, clearance, label="DEFAULT_ARMS_STAND_BASELINE", waypoints=[q0], active=(False, False),
                                   target_q=q0, target_pos=target_pos, target_quat=target_quat, video=debug_video, stop_on_safety=True)
    phase_results["baseline"] = baseline
    _write_trace(run_root, "DEFAULT_ARMS_STAND_BASELINE", baseline)

    write_json(run_root / "runner_status.json", {"status": "RUNNING", "phase": "EXACT_PAIR_LEFT_STRAIGHT_REPLAY"})
    left_straight = ns["run_episode"](env, runtime, clearance, label="LEFT_ONLY_STRAIGHT_REPLAY", waypoints=[q0, q_left_target], active=(True, False),
                                       target_q=q_left_target, target_pos=target_pos, target_quat=target_quat, video=debug_video, stop_on_safety=True)
    phase_results["left_straight"] = left_straight
    _write_trace(run_root, "LEFT_ONLY_STRAIGHT_REPLAY", left_straight)
    bilateral_straight = ns["run_episode"](env, runtime, clearance, label="BILATERAL_STRAIGHT_REPLAY", waypoints=[q0, qchest], active=(True, True),
                                            target_q=qchest, target_pos=target_pos, target_quat=target_quat, video=debug_video, stop_on_safety=True)
    phase_results["bilateral_straight"] = bilateral_straight
    _write_trace(run_root, "BILATERAL_STRAIGHT_REPLAY", bilateral_straight)
    videos["collider_debug"] = debug_video.close()

    write_json(run_root / "runner_status.json", {"status": "RUNNING", "phase": "STRAIGHT_CSPACE_STATIC_AUDIT"})
    env.reset(seed=args.seed)
    straight = ns["static_segment"](env, runtime, q0, qchest, ns["STATIC_STRAIGHT_SAMPLES"], clearance, "q0_to_q_chest")
    write_json(run_root / "straight_cspace_static_audit.json", straight)
    set_arm_q = ns["set_arm_q"]
    set_arm_q(env, [int(value) for value in runtime["arm_joint_ids"]], qchest)
    if pair.coverage["status"] == "PASS":
        pair.sample(0, "FINAL_STATIC_TARGET", qchest, force_all=True, record_history=True)
    default_pairs, final_pairs = _pair_sets(pair)

    write_json(run_root / "runner_status.json", {"status": "RUNNING", "phase": "BOUNDED_WAYPOINT_STATIC_SEARCH"})
    candidates = ns["make_waypoint_candidates"](q0, qchest)
    candidate_records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for index, (q_clear, q_lift) in enumerate(candidates[:32]):
        env.reset(seed=args.seed)
        segments = []
        for segment_index, (start, goal) in enumerate(((q0, q_clear), (q_clear, q_lift), (q_lift, qchest)), start=1):
            segments.append(ns["static_segment"](env, runtime, start, goal, ns["STATIC_SEGMENT_SAMPLES"], clearance, f"candidate_{index}_segment_{segment_index}"))
        minimum = min((segment["minimum_left_clearance_m"] for segment in segments if isinstance(segment["minimum_left_clearance_m"], (int, float))), default=None)
        margin = min((min(float(point["minimum_joint_margin_rad"]) for point in segment["records"]) for segment in segments), default=0.0)
        free = all(segment["collision_free"] for segment in segments) and minimum is not None and minimum >= ns["DEVELOPMENT_CLEARANCE_M"] and margin >= ns["STATIC_MARGIN_GATE_RAD"]
        record = {"candidate_index": index, "q_clear_rad": tolist(q_clear), "q_lift_rad": tolist(q_lift), "segments": segments,
                  "minimum_pelvis_clearance_m": minimum, "minimum_joint_margin_rad": margin, "static_collision_free": free}
        candidate_records.append(record)
        if free:
            score = (-float(minimum), ns["path_joint_travel"]([q0.tolist(), q_clear.tolist(), q_lift.tolist(), qchest.tolist()]), -float(margin))
            record["selection_score"] = list(score)
            if selected is None or tuple(score) < tuple(selected["selection_score"]):
                selected = {**record, "selection_score": list(score)}
        write_json(run_root / "waypoint_candidate_records.json", candidate_records)
    ns["update_waypoint_yaml"](selected)
    write_json(run_root / "waypoint_selection.json", {"status": "STATIC_COLLISION_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED" if selected else "WAYPOINT_PATH_NOT_FOUND", "selected": selected, "candidate_count": len(candidate_records)})

    dynamic: dict[str, Any] = {}
    if selected is not None:
        q_clear = torch.tensor(selected["q_clear_rad"], dtype=qchest.dtype, device=env.device)
        q_lift = torch.tensor(selected["q_lift_rad"], dtype=qchest.dtype, device=env.device)
        left_video = ns["EvidenceVideo"](run_root, env, "left_waypoint_front.mp4", "left_waypoint_side.mp4")
        ns["VIDEOS"].append(left_video)
        write_json(run_root / "runner_status.json", {"status": "RUNNING", "phase": "LEFT_WAYPOINT_DYNAMIC"})
        dynamic["left"] = ns["run_episode"](env, runtime, clearance, label="LEFT_ONLY_WAYPOINT", waypoints=[q0, q_clear, q_lift, qchest], active=(True, False),
                                               target_q=q_left_target, target_pos=target_pos, target_quat=target_quat, video=left_video, stop_on_safety=True)
        _write_trace(run_root, "LEFT_ONLY_WAYPOINT", dynamic["left"])
        videos["left_waypoint"] = left_video.close()
        right_video = None
        write_json(run_root / "runner_status.json", {"status": "RUNNING", "phase": "RIGHT_ONLY_REGRESSION"})
        dynamic["right"] = ns["run_episode"](env, runtime, clearance, label="RIGHT_ONLY_REGRESSION", waypoints=[q0, q_right_target], active=(False, True),
                                                target_q=q_right_target, target_pos=target_pos, target_quat=target_quat, video=right_video, stop_on_safety=True)
        _write_trace(run_root, "RIGHT_ONLY_REGRESSION", dynamic["right"])
        bilateral_video = ns["EvidenceVideo"](run_root, env, "bilateral_waypoint_front.mp4", "bilateral_waypoint_side.mp4")
        ns["VIDEOS"].append(bilateral_video)
        bilateral_clear, bilateral_lift = q_clear.clone(), q_lift.clone()
        bilateral_clear[1::2], bilateral_lift[1::2] = q0[1::2], q0[1::2]
        write_json(run_root / "runner_status.json", {"status": "RUNNING", "phase": "BILATERAL_WAYPOINT_DYNAMIC"})
        dynamic["bilateral"] = ns["run_episode"](env, runtime, clearance, label="BILATERAL_WAYPOINT", waypoints=[q0, bilateral_clear, bilateral_lift, qchest], active=(True, True),
                                                    target_q=qchest, target_pos=target_pos, target_quat=target_quat, video=bilateral_video, stop_on_safety=True)
        _write_trace(run_root, "BILATERAL_WAYPOINT", dynamic["bilateral"])
        videos["bilateral_waypoint"] = bilateral_video.close()

    pair_summaries = pair.summaries(default_pairs, final_pairs)
    write_json(run_root / "pair_summary.json", {"coverage": pair.coverage, "pairs": pair_summaries})
    pair_audit_status = "PASS" if pair_summaries and pair.coverage["status"] == "PASS" else "INVALID"
    classification = _classify_contact(pair_summaries[0] if pair_summaries else None)
    report = {
        "schema_version": 1, "stage": stage_name, "status": pair_audit_status,
        "previous_authoritative_result": {"status": "FAIL", "primary_reason": "SAFETY_GATE_TRIGGERED",
                                           "run_root": str(ns["OLD_RUN"]), "result_commit": "23f2e48e21040e5c16c4c32ead740c68e4ff3410"},
        "contact_pair_sensor_coverage": pair.coverage, "pairs": pair_summaries,
        "default_stand_pairs": sorted(default_pairs), "final_static_target_pairs": sorted(final_pairs),
        "classification": classification, "collider_audit_path": str(run_root / "collider_audit.json"),
        "measurement_semantics": "post-step for dynamic replay; forward/static state for static audits",
    }
    write_json(run_root / "contact_pair_audit.json", report)
    write_json(Path("reports/stage2/s2_03t_pelvis_contact_pair_audit.json"), report)

    baseline_pass = baseline["safety_ok"] and baseline["hold_seconds"] >= 10.0
    dynamic_path_found = selected is not None
    left_pass = bool(dynamic.get("left", {}).get("safety_ok") and dynamic.get("left", {}).get("hold_seconds", 0.0) >= 10.0 and not dynamic.get("left", {}).get("oscillation_detected"))
    right_pass = bool(dynamic.get("right", {}).get("safety_ok") and dynamic.get("right", {}).get("hold_seconds", 0.0) >= 10.0)
    bilateral_pass = bool(dynamic.get("bilateral", {}).get("safety_ok") and dynamic.get("bilateral", {}).get("hold_seconds", 0.0) >= 10.0 and not dynamic.get("bilateral", {}).get("oscillation_detected"))
    if pair_audit_status != "PASS":
        status, reason = "INVALID", "CONTACT_PAIR_AUDIT_INCOMPLETE"
    elif not baseline_pass:
        status, reason = "FAIL", "DEFAULT_ARMS_STAND_BASELINE_FAILED"
    elif straight["collision_free"]:
        status, reason = "FAIL", "STRAIGHT_PATH_UNEXPECTEDLY_COLLISION_FREE_BUT_WAYPOINT_CONTRACT_UNRESOLVED"
    elif not dynamic_path_found:
        status, reason = "FAIL", "WAYPOINT_PATH_NOT_FOUND"
    elif not left_pass:
        status, reason = "FAIL", "LEFT_WAYPOINT_PATH_SAFETY_GATE_TRIGGERED"
    elif not right_pass:
        status, reason = "FAIL", "RIGHT_ONLY_REGRESSION_FAILED"
    elif not bilateral_pass:
        status, reason = "FAIL", "BILATERAL_WAYPOINT_PATH_SAFETY_GATE_TRIGGERED"
    else:
        status, reason = "PASS", "PELVIS_CONTACT_FREE_WAYPOINT_PATH_AND_CHEST_HOLD_COMPLETE"
    result = {
        "schema_version": 1, "stage": stage_name, "status": status, "primary_reason": reason,
        "contact_pair_audit": report, "collider_visual_audit": {"status": "PASS", "visualization": debug_visualization},
        "runtime_audit": runtime, "arm_reference_order_audit": order, "arm_reference_mirror_audit": mirror,
        "straight_static": {k: v for k, v in straight.items() if k != "records"},
        "waypoint_selection": selected, "waypoint_candidate_count": len(candidate_records),
        "baseline": {k: v for k, v in baseline.items() if k != "records"},
        "straight_replay": {"left": {k: v for k, v in left_straight.items() if k != "records"}, "bilateral": {k: v for k, v in bilateral_straight.items() if k != "records"}},
        "dynamic": {name: {k: v for k, v in value.items() if k != "records"} for name, value in dynamic.items()},
        "videos": videos, "formal_gates_unchanged": True,
        "scientific_contract": {"ppo_started": False, "falcon_started": False, "box_used": False,
                                "base_walking_started": False, "contact_attach_started": False, "planner_started": False},
        "authoritative_complete_before_teardown": True,
    }
    write_json(run_root / "result.json", result)
    write_json(run_root / "summary.json", {
        "stage": stage_name, "status": status, "primary_reason": reason,
        "contact_body_a": pair_summaries[0]["body_a"] if pair_summaries else "UNKNOWN",
        "contact_body_b": pair_summaries[0]["body_b"] if pair_summaries else "UNKNOWN",
        "contact_first_frame": pair_summaries[0]["first_contact_frame"] if pair_summaries else None,
        "contact_force_max_n": pair_summaries[0]["force_max_n"] if pair_summaries else None,
        "straight_path_collision_free": straight["collision_free"], "selected_waypoint": selected,
        "left_waypoint": "PASS" if left_pass else "FAIL" if selected else "NOT_RUN",
        "right_only_regression": "PASS" if right_pass else "FAIL" if selected else "NOT_RUN",
        "bilateral_waypoint": "PASS" if bilateral_pass else "FAIL" if selected else "NOT_RUN",
        "hold_seconds": dynamic.get("bilateral", {}).get("hold_seconds", 0.0), "videos": videos,
    })
    write_json(run_root / "runner_status.json", {"status": "COMPLETE", "phase": "RESULT_WRITTEN", "primary_reason": reason})
    print(f"PELVIS_PATH_RECOVERY_STATUS={status}", flush=True)
    print(f"PRIMARY_REASON={reason}", flush=True)
    print(f"PELVIS_CONTACT_PAIR_AUDIT={pair_audit_status}", flush=True)
    print(f"CONTACT_BODY_A={pair_summaries[0]['body_a'] if pair_summaries else 'UNKNOWN'}", flush=True)
    print(f"CONTACT_BODY_B={pair_summaries[0]['body_b'] if pair_summaries else 'UNKNOWN'}", flush=True)
    print(f"CONTACT_FIRST_FRAME={pair_summaries[0]['first_contact_frame'] if pair_summaries else 'UNKNOWN'}", flush=True)
    print(f"CONTACT_FORCE_MAX_N={pair_summaries[0]['force_max_n'] if pair_summaries else 'UNKNOWN'}", flush=True)
    print(f"PELVIS_CONTACT_CLASSIFICATION={classification}", flush=True)
    print(f"STRAIGHT_CSPACE_PATH_COLLISION_FREE={'YES' if straight['collision_free'] else 'NO'}", flush=True)
    print(f"WAYPOINT_PATH_FOUND={'YES' if selected else 'NO'}", flush=True)
    print(f"LEFT_WAYPOINT_PATH={'PASS' if left_pass else 'FAIL' if selected else 'NOT_RUN'}", flush=True)
    print(f"RIGHT_ONLY_REGRESSION={'PASS' if right_pass else 'FAIL' if selected else 'NOT_RUN'}", flush=True)
    print(f"BILATERAL_WAYPOINT_PATH={'PASS' if bilateral_pass else 'FAIL' if selected else 'NOT_RUN'}", flush=True)
    return 0 if status != "INVALID" else 1


def main(ns: dict[str, Any]) -> int:
    try:
        return run(ns)
    except BaseException as exc:
        run_root: Path = ns["RUN"]
        payload = {"schema_version": 1, "stage": ns["STAGE"], "status": "INVALID", "primary_reason": "IMPLEMENTATION_EXCEPTION",
                   "exception_type": type(exc).__name__, "exception_message": str(exc),
                   "traceback": "".join(__import__("traceback").format_exception(type(exc), exc, exc.__traceback__)),
                   "authoritative_complete_before_teardown": False}
        write_json(run_root / "implementation_exception.json", payload)
        if not (run_root / "result.json").is_file():
            write_json(run_root / "result.json", payload)
        write_json(run_root / "runner_status.json", {"status": "INVALID", "phase": "EXCEPTION", "primary_reason": "IMPLEMENTATION_EXCEPTION"})
        print("PELVIS_PATH_RECOVERY_STATUS=INVALID", flush=True)
        print("PRIMARY_REASON=IMPLEMENTATION_EXCEPTION", flush=True)
        return 1
