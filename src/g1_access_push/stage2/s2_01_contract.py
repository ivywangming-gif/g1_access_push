"""Pure S2-01 baseline resolution and PASS/FAIL/INVALID classification."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

REQUIRED_AUDIT_FIELDS = {
    "box_mass_properties_audit.json": {
        "body_prim_path", "authored_mass_kg", "runtime_mass_kg",
        "authored_com_local_xyz_m", "runtime_com_local_xyz_m",
        "runtime_com_world_xyz_m", "authored_diagonal_inertia_kg_m2",
        "runtime_diagonal_inertia_kg_m2", "authored_principal_axes_wxyz",
        "runtime_principal_axes_wxyz", "tolerance_checks", "low_com_pass",
    },
    "physics_material_audit.json": {
        "box_material_prim_path", "ground_material_prim_path",
        "box_binding_target", "ground_binding_target", "box", "ground",
        "effective_pair", "pair_pass",
    },
    "scene_geometry_audit.json": {
        "robot_max_x_world_m", "box_rear_face_x_world_m",
        "box_center_xyz_world_m", "expected_clearance_m",
        "measured_minimum_clearance_m", "overlap_count",
        "scene_query_robot_hit", "doorway_count", "obstacle_count",
    },
    "box_rigid_body_audit.json": {
        "body_prim_path", "cube_collider_prim_path", "runtime_size_xyz_m",
        "rigid_body_enabled", "kinematic_enabled", "gravity_enabled",
    },
    "contact_sensor_audit.json": {
        "box_net_sensor_prim_path", "box_robot_sensor_prim_path",
        "box_net_body_names", "box_robot_body_names", "box_robot_filter_paths",
        "net_forces_available", "robot_force_matrix_available",
        "contact_reporter_initialized", "sensor_audit_pass",
        "configured_prim_path", "resolved_prim_expression",
        "resolved_body_names", "num_bodies", "rigid_body_bound",
        "configured_filter_patterns", "configured_filter_pattern_count",
        "backend_filter_count", "filter_semantics", "force_matrix_shape",
        "force_matrix_m", "force_matrix_m_matches_backend_filter_count",
        "usd_candidate_robot_rigid_body_count", "usd_body_count_used_as_shape_contract",
        "net_force_shape", "net_force_finite", "net_force_xyz_n",
        "net_force_norm_n", "filter_tensor_initialization_pass",
    },
}

VALID_FAIL_REASONS = (
    "BOX_GEOMETRY_MISMATCH", "BOX_NOT_DYNAMIC", "BOX_GRAVITY_DISABLED",
    "BOX_MASS_MISMATCH", "BOX_LOW_COM_CONTRACT_NOT_SATISFIED",
    "BOX_INERTIA_AUDIT_FAILED", "PHYSICS_MATERIAL_MISMATCH",
    "INITIAL_ROBOT_BOX_OVERLAP", "INITIAL_CLEARANCE_MISMATCH",
    "UNEXPECTED_ROBOT_BOX_CONTACT", "FORBIDDEN_BODY_BOX_COLLISION",
    "OBJECT_TRANSLATION_WITHOUT_CONTACT", "OBJECT_YAW_DRIFT",
    "OBJECT_ROLL_PITCH_DRIFT", "OBJECT_VERTICAL_DRIFT",
    "OBJECT_LINEAR_SPEED_EXCEEDED", "OBJECT_ANGULAR_SPEED_EXCEEDED",
    "ROBOT_FALL", "ROBOT_BAD_TILT", "ROBOT_ROOT_HEIGHT_VIOLATION",
    "NONFINITE", "AUTO_RESET_DETECTED", "FRAME_COUNT_MISMATCH",
    "GROUND_SUPPORT_CONTACT_MISSING",
)

INVALID_REASONS = (
    "IMPLEMENTATION_EXCEPTION", "MISSING_RESULT_JSON", "MISSING_TRACE",
    "MISSING_FINAL_IMAGE", "MISSING_REQUIRED_FIELD",
    "STANDING_ACTION_CONTRACT_UNCERTIFIED", "COLLISION_BACKEND_UNCERTIFIED",
    "ROBOT_RUNTIME_ENVELOPE_QUERY_FAILED", "MASS_PROPERTY_QUERY_FAILED",
    "MATERIAL_BINDING_QUERY_FAILED", "MULTIPLE_ISAAC_PROCESSES",
    "EVALUATOR_DID_NOT_COMPLETE", "CONTACT_SENSOR_INITIALIZATION_FAILED",
    "CONTACT_SENSOR_AUDIT_IMPLEMENTATION_ERROR",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ballast_inertia(mass: float, size_xyz: Iterable[float]) -> list[float]:
    x, y, z = [float(value) for value in size_xyz]
    return [
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    ]


def placement_from_robot_max_x(
    robot_max_x_world_m: float,
    clearance_m: float,
    half_length_m: float,
    center_y_world_m: float,
    center_z_world_m: float,
) -> dict[str, Any]:
    rear = robot_max_x_world_m + clearance_m
    return {
        "rear_face_x_world_m": rear,
        "center_xyz_world_m": [rear + half_length_m, center_y_world_m, center_z_world_m],
        "measured_minimum_clearance_m": rear - robot_max_x_world_m,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["baseline"]["baseline_id"] != "S2_LIGHT_BOX_DEVELOPMENT_BASELINE_V1":
        raise ValueError("unexpected development baseline")
    mass = float(config["mass_properties"]["mass_kg"])
    computed = ballast_inertia(mass, config["mass_properties"]["ballast_equivalent_size_xyz_m"])
    expected = config["mass_properties"]["diagonal_inertia_kg_m2"]
    if any(abs(a - b) > 1.0e-12 for a, b in zip(computed, expected)):
        raise ValueError("ballast inertia formula mismatch")
    if config["evaluation"]["expected_frames"] != 3000:
        raise ValueError("S2-01 requires 3000 frames")
    if any(config["prohibitions"].values()):
        raise ValueError("forbidden S2-01 capability enabled")


def resolved_config(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "runnable": True,
        "qualification_state": "READY_FOR_SINGLE_FORMAL_RUN",
        "source_config_path": str(config_path.resolve()),
        "source_config_sha256": sha256_file(config_path),
        "contract_digest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "resolved": config,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_summary(records: list[dict[str, Any]], key: str, threshold: float) -> dict[str, Any]:
    values = [float(record[key]) for record in records]
    first = next((int(record["frame"]) for record in records if float(record[key]) > threshold), None)
    return {
        "maximum": max(values) if values else math.nan,
        "p95": percentile95(values),
        "final": values[-1] if values else math.nan,
        "threshold": threshold,
        "first_threshold_exceeding_frame": first,
        "pass": first is None,
    }


def classify_evidence(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    *,
    runner_rc: int,
    final_image_present: bool,
    multiple_isaac_processes: bool,
) -> dict[str, Any]:
    invalid: list[str] = []
    failures: list[str] = []
    expected = int(config["evaluation"]["expected_frames"])
    if runner_rc != 0:
        invalid.append("IMPLEMENTATION_EXCEPTION")
    if multiple_isaac_processes:
        invalid.append("MULTIPLE_ISAAC_PROCESSES")
    if not final_image_present:
        invalid.append("MISSING_FINAL_IMAGE")
    if len(records) != expected:
        failures.append("FRAME_COUNT_MISMATCH")
    for name, required in REQUIRED_AUDIT_FIELDS.items():
        if name not in audits:
            invalid.append("MISSING_REQUIRED_FIELD")
        elif not required.issubset(audits[name]):
            invalid.append("MISSING_REQUIRED_FIELD")
    if "contact_sensor_audit.json" in audits and not audits["contact_sensor_audit.json"].get("sensor_audit_pass", False):
        invalid.append("CONTACT_SENSOR_INITIALIZATION_FAILED")
    if "physics_material_audit.json" in audits:
        material = audits["physics_material_audit.json"]
        if not material.get("box_binding_target") or not material.get("ground_binding_target"):
            invalid.append("MATERIAL_BINDING_QUERY_FAILED")
    if invalid:
        return {"status": "INVALID", "primary_reason": invalid[0], "all_reasons": invalid}

    mass_audit = audits["box_mass_properties_audit.json"]
    material_audit = audits["physics_material_audit.json"]
    geometry_audit = audits["scene_geometry_audit.json"]
    rigid_audit = audits["box_rigid_body_audit.json"]
    if max(abs(float(rigid_audit["runtime_size_xyz_m"][i]) - float(config["geometry"]["size_xyz_m"][i])) for i in range(3)) > 1.0e-5:
        failures.append("BOX_GEOMETRY_MISMATCH")
    if not rigid_audit["rigid_body_enabled"] or rigid_audit["kinematic_enabled"]:
        failures.append("BOX_NOT_DYNAMIC")
    if not rigid_audit["gravity_enabled"]:
        failures.append("BOX_GRAVITY_DISABLED")
    if not mass_audit["tolerance_checks"].get("mass", False):
        failures.append("BOX_MASS_MISMATCH")
    if not mass_audit.get("low_com_pass", False):
        failures.append("BOX_LOW_COM_CONTRACT_NOT_SATISFIED")
    if not mass_audit["tolerance_checks"].get("inertia", False):
        failures.append("BOX_INERTIA_AUDIT_FAILED")
    if not material_audit.get("pair_pass", False):
        failures.append("PHYSICS_MATERIAL_MISMATCH")
    if int(geometry_audit["overlap_count"]) != 0 or geometry_audit["scene_query_robot_hit"]:
        failures.append("INITIAL_ROBOT_BOX_OVERLAP")
    if float(geometry_audit["measured_minimum_clearance_m"]) < float(config["placement"]["minimum_allowed_clearance_m"]):
        failures.append("INITIAL_CLEARANCE_MISMATCH")

    if records:
        finite = all(bool(record["finite"]) for record in records)
        if not finite:
            failures.append("NONFINITE")
        if any(bool(record["robot_box_contact"]) for record in records):
            failures.append("UNEXPECTED_ROBOT_BOX_CONTACT")
        if any(int(record.get("runtime_forbidden_overlap_count", 0)) > 0 for record in records):
            failures.append("FORBIDDEN_BODY_BOX_COLLISION")
        if any(bool(record["robot_fall"]) for record in records):
            failures.append("ROBOT_FALL")
        if any(bool(record["robot_bad_tilt"]) for record in records):
            failures.append("ROBOT_BAD_TILT")
        if any(float(record["root_height_m"]) < config["robot"]["fall_root_height_threshold_m"] for record in records):
            failures.append("ROBOT_ROOT_HEIGHT_VIOLATION")
        if any(int(record["post_initial_reset_count"]) != 0 for record in records):
            failures.append("AUTO_RESET_DETECTED")

    post = [record for record in records if int(record["frame"]) >= config["evaluation"]["first_frame"]]
    thresholds = config["evaluation"]["stillness_thresholds"]
    metric_reason = {
        "box_linear_speed_mps": ("max_box_linear_speed_mps", "OBJECT_LINEAR_SPEED_EXCEEDED"),
        "box_angular_speed_radps": ("max_box_angular_speed_radps", "OBJECT_ANGULAR_SPEED_EXCEEDED"),
        "box_translation_from_reference_m": ("max_box_translation_from_reference_m", "OBJECT_TRANSLATION_WITHOUT_CONTACT"),
        "box_abs_yaw_change_rad": ("max_box_yaw_change_rad", "OBJECT_YAW_DRIFT"),
        "box_abs_roll_rad": ("max_box_abs_roll_rad", "OBJECT_ROLL_PITCH_DRIFT"),
        "box_abs_pitch_rad": ("max_box_abs_pitch_rad", "OBJECT_ROLL_PITCH_DRIFT"),
        "box_abs_vertical_drift_m": ("max_box_vertical_drift_m", "OBJECT_VERTICAL_DRIFT"),
    }
    metrics: dict[str, Any] = {}
    for metric, (threshold_key, reason) in metric_reason.items():
        metrics[metric] = metric_summary(post, metric, float(thresholds[threshold_key]))
        if not metrics[metric]["pass"]:
            failures.append(reason)
    if post and not all(float(record["box_ground_contact_force_n"]) > 0.0 for record in post):
        failures.append("GROUND_SUPPORT_CONTACT_MISSING")
    unique = list(dict.fromkeys(failures))
    return {
        "status": "FAIL" if unique else "PASS",
        "primary_reason": unique[0] if unique else "ALL_FROZEN_GATES_PASSED",
        "all_reasons": unique,
        "stillness_metrics": metrics,
    }
