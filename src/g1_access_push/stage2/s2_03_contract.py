"""Pure S2-03 parameter derivation, rate limiting, and result classification."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from g1_access_push.stage2.contract import PHYSICAL_FAILURE_REASONS

INVALID_REASONS = {
    "IMPLEMENTATION_EXCEPTION", "MISSING_REQUIRED_FIELD", "MISSING_TRACE",
    "NONFINITE", "CONFIG_SHA_MISMATCH", "CHECKPOINT_CONTRACT_FAILED",
    "MULTIPLE_ISAAC_PROCESSES", "MISSING_FINAL_IMAGE", "FSM_EVIDENCE_MISMATCH",
}


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("stage") != "S2-03":
        raise ValueError("not an S2-03 config")
    if data["formal"]["episode_count"] != 1 or data["formal"]["auto_reset"] is not False:
        raise ValueError("S2-03 requires one non-resetting episode")
    if any(data["prohibitions"][key] for key in ("pushing", "planner", "training", "forbidden_pilot_checkpoint")):
        raise ValueError("forbidden S2-03 behavior enabled")
    return data


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def derive_frozen_parameters(s2_01_audit: dict[str, Any], s2_02_result: dict[str, Any], s2_01_config: dict[str, Any]) -> dict[str, Any]:
    noise_s2_01 = max(
        float(s2_01_audit["initial_robot_contact_force_max_n"]),
        float(s2_01_audit["episode_robot_contact_force_max_n"]),
    )
    noise_s2_02 = float(s2_02_result["metrics"]["robot_box_contact_force_n"]["maximum"])
    noise_upper = max(noise_s2_01, noise_s2_02)
    frequency = float(s2_01_config["simulation"]["control_frequency_hz"])
    dt = 1.0 / frequency
    mass = float(s2_01_audit["mass_kg"])
    static_friction = float(s2_01_config["materials"]["effective_pair"]["static_friction"])
    friction_capacity = static_friction * mass * 9.81
    per_palm_peak = 0.4 * friction_capacity
    return {
        "control_frequency_hz": frequency,
        "control_dt_s": dt,
        "no_contact_noise_upper_bound_n": noise_upper,
        "contact_force_threshold_n": max(1.0, noise_upper + 1.0),
        "precontact_gap_m": float(s2_02_result["selected_candidate"]["precontact_gap_m"]),
        "approach_speed_mps": 0.01,
        "approach_acceleration_limit_mps2": 0.02,
        "approach_jerk_limit_mps3": 0.10,
        "bilateral_verification_steps": round(0.40 * frequency),
        "attached_hold_steps": round(2.0 * frequency),
        "contact_loss_grace_steps": round(0.10 * frequency),
        "single_hand_maximum_steps": round(0.50 * frequency),
        "contact_impulse_window_steps": round(0.10 * frequency),
        "static_friction_capacity_n": friction_capacity,
        "per_palm_force_peak_threshold_n": per_palm_peak,
        "per_palm_impulse_threshold_ns": per_palm_peak * 0.10,
        "excessive_combined_impulse_threshold_ns": 2.0 * per_palm_peak * 0.10,
        "contact_force_rate_threshold_nps": per_palm_peak / dt,
        "maximum_box_linear_speed_mps": float(s2_01_config["evaluation"]["stillness_thresholds"]["max_box_linear_speed_mps"]),
        "maximum_box_angular_speed_radps": float(s2_01_config["evaluation"]["stillness_thresholds"]["max_box_angular_speed_radps"]),
        "maximum_box_translation_m": float(s2_01_config["evaluation"]["stillness_thresholds"]["max_box_translation_from_reference_m"]),
        "maximum_box_yaw_change_rad": float(s2_01_config["evaluation"]["stillness_thresholds"]["max_box_yaw_change_rad"]),
    }


def advance_rate_limited(
    displacement_m: float,
    speed_mps: float,
    acceleration_mps2: float,
    *,
    dt_s: float,
    speed_limit_mps: float,
    acceleration_limit_mps2: float,
    jerk_limit_mps3: float,
) -> tuple[float, float, float]:
    target_acceleration = acceleration_limit_mps2 if speed_mps < speed_limit_mps else 0.0
    delta_limit = jerk_limit_mps3 * dt_s
    acceleration_mps2 += max(-delta_limit, min(delta_limit, target_acceleration - acceleration_mps2))
    acceleration_mps2 = max(0.0, min(acceleration_limit_mps2, acceleration_mps2))
    speed_mps = min(speed_limit_mps, speed_mps + acceleration_mps2 * dt_s)
    displacement_m += speed_mps * dt_s
    return displacement_m, speed_mps, acceleration_mps2


def resolved_config(config_path: Path, s2_01_audit_path: Path, s2_02_result_path: Path, s2_01_config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    sources = {
        "s2_01_runtime_audit": json.loads(s2_01_audit_path.read_text(encoding="utf-8")),
        "s2_02_formal_result": json.loads(s2_02_result_path.read_text(encoding="utf-8")),
        "s2_01_config": yaml.safe_load(s2_01_config_path.read_text(encoding="utf-8")),
    }
    derived = derive_frozen_parameters(sources["s2_01_runtime_audit"], sources["s2_02_formal_result"], sources["s2_01_config"])
    checks = {
        "s2_01_pass": sources["s2_01_runtime_audit"].get("status") == "PASS",
        "s2_02_pass": sources["s2_02_formal_result"].get("status") == "PASS",
        "control_frequency": math.isclose(config["controller"]["control_frequency_hz"], derived["control_frequency_hz"]),
        "control_dt": math.isclose(config["controller"]["control_dt_s"], derived["control_dt_s"]),
        "precontact_gap": math.isclose(config["precontact"]["precontact_gap_m"], derived["precontact_gap_m"]),
        "contact_threshold": math.isclose(config["contact_gate"]["left_contact_force_threshold_n"], derived["contact_force_threshold_n"]) and math.isclose(config["contact_gate"]["right_contact_force_threshold_n"], derived["contact_force_threshold_n"]),
        "motion_limits": all(math.isclose(config["motion"][key], derived[key]) for key in ("approach_speed_mps", "approach_acceleration_limit_mps2", "approach_jerk_limit_mps3")),
        "timing": all(config["motion"][key] == derived[key] for key in ("bilateral_verification_steps", "attached_hold_steps", "contact_loss_grace_steps", "single_hand_maximum_steps", "contact_impulse_window_steps")),
        "force_safety": all(math.isclose(config["contact_gate"][key], derived[key]) for key in ("per_palm_force_peak_threshold_n", "per_palm_impulse_threshold_ns", "excessive_combined_impulse_threshold_ns", "contact_force_rate_threshold_nps")),
        "box_stillness": all(math.isclose(config["acceptance"][key], derived[key]) for key in ("maximum_box_linear_speed_mps", "maximum_box_angular_speed_radps", "maximum_box_translation_m", "maximum_box_yaw_change_rad")),
    }
    provenance = {
        "schema_version": 1, "stage": "S2-03", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "derived_parameters": derived,
        "source_sha256": {
            str(s2_01_audit_path): sha256_file(s2_01_audit_path),
            str(s2_02_result_path): sha256_file(s2_02_result_path),
            str(s2_01_config_path): sha256_file(s2_01_config_path),
        },
        "rules": {
            "contact_threshold": "max(1 N, max(S2-01 noise, S2-02 noise) + 1 N)",
            "approach": "0.01 m/s at 50 Hz; acceleration=2*speed; jerk=5*acceleration",
            "force_peak": "each palm <= 0.4 * mu_static * mass * g; bilateral total <= 0.8 friction capacity",
            "impulse": "per-palm peak * 0.10 s; combined is twice per-palm",
            "box_stillness": "unchanged S2-01/S2-02 frozen stillness gates",
        },
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    resolved = {
        "schema_version": 1, "stage": "S2-03", "runnable": all(checks.values()),
        "qualification_state": "READY_FOR_PREFLIGHT_AND_FORMAL" if all(checks.values()) else "PARAMETER_PROVENANCE_FAILED",
        "config_sha256": sha256_file(config_path),
        "contract_digest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "resolved": config, "parameter_provenance": provenance,
    }
    return resolved, provenance


def classify_formal(config: dict[str, Any], records: list[dict[str, Any]], outcome: dict[str, Any], *, final_image_present: bool) -> dict[str, Any]:
    if not records:
        return {"status": "INVALID", "primary_reason": "MISSING_TRACE", "all_reasons": ["MISSING_TRACE"]}
    if not final_image_present:
        return {"status": "INVALID", "primary_reason": "MISSING_FINAL_IMAGE", "all_reasons": ["MISSING_FINAL_IMAGE"]}
    required = {
        "frame", "fsm_state", "finite", "left_contact", "right_contact", "left_force_n", "right_force_n",
        "left_impulse_ns", "right_impulse_ns", "combined_impulse_ns", "contact_force_peak_n", "contact_force_rate_nps",
        "forbidden_contact_links", "box_translation_m", "box_yaw_change_rad", "box_linear_speed_mps",
        "box_angular_speed_radps", "root_height_m", "root_tilt_deg", "minimum_arm_joint_limit_margin_rad",
        "arm_torque_ratio_max", "post_initial_reset_count",
    }
    if any(not required.issubset(record) for record in records):
        return {"status": "INVALID", "primary_reason": "MISSING_REQUIRED_FIELD", "all_reasons": ["MISSING_REQUIRED_FIELD"]}
    if not all(record["finite"] for record in records):
        return {"status": "INVALID", "primary_reason": "NONFINITE", "all_reasons": ["NONFINITE"]}
    terminal = outcome.get("terminal_state")
    failure_reason = outcome.get("failure_reason")
    if terminal == "FAIL":
        if failure_reason not in PHYSICAL_FAILURE_REASONS:
            return {"status": "INVALID", "primary_reason": "FSM_EVIDENCE_MISMATCH", "all_reasons": ["FSM_EVIDENCE_MISMATCH"]}
        return {"status": "FAIL", "primary_reason": failure_reason, "all_reasons": [failure_reason]}
    if terminal != "PASS":
        return {"status": "INVALID", "primary_reason": "FSM_EVIDENCE_MISMATCH", "all_reasons": ["FSM_EVIDENCE_MISMATCH"]}
    states = {record["fsm_state"] for record in records}
    if not {"STAND_SETTLE", "PRECONTACT", "APPROACH_NORMAL", "BILATERAL_CONTACT_VERIFY", "ATTACHED_HOLD"}.issubset(states):
        return {"status": "INVALID", "primary_reason": "FSM_EVIDENCE_MISMATCH", "all_reasons": ["FSM_EVIDENCE_MISMATCH"]}
    hold = [record for record in records if record["fsm_state"] == "ATTACHED_HOLD"]
    if len(hold) < int(config["motion"]["attached_hold_steps"]):
        return {"status": "INVALID", "primary_reason": "FSM_EVIDENCE_MISMATCH", "all_reasons": ["FSM_EVIDENCE_MISMATCH"]}
    acceptance = config["acceptance"]
    gates = [
        all(record["left_contact"] and record["right_contact"] for record in hold),
        not any(record["forbidden_contact_links"] for record in records),
        max(record["box_translation_m"] for record in hold) <= acceptance["maximum_box_translation_m"],
        max(record["box_yaw_change_rad"] for record in hold) <= acceptance["maximum_box_yaw_change_rad"],
        max(record["box_linear_speed_mps"] for record in hold) <= acceptance["maximum_box_linear_speed_mps"],
        max(record["box_angular_speed_radps"] for record in hold) <= acceptance["maximum_box_angular_speed_radps"],
        max(record["root_tilt_deg"] for record in records) <= acceptance["maximum_root_tilt_deg"],
        min(record["root_height_m"] for record in records) >= acceptance["minimum_root_height_m"],
        max(record["root_height_m"] for record in records) <= acceptance["maximum_root_height_m"],
        min(record["minimum_arm_joint_limit_margin_rad"] for record in records) >= acceptance["minimum_arm_joint_limit_margin_rad"],
        max(record["arm_torque_ratio_max"] for record in records) <= acceptance["maximum_arm_torque_ratio"],
        max(record["post_initial_reset_count"] for record in records) == 0,
    ]
    if not all(gates):
        return {"status": "INVALID", "primary_reason": "FSM_EVIDENCE_MISMATCH", "all_reasons": ["FSM_EVIDENCE_MISMATCH"]}
    return {"status": "PASS", "primary_reason": "ALL_S2_03_ATTACH_GATES_PASSED", "all_reasons": []}
