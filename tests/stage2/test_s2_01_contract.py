"""Pure static tests for the frozen S2-01 development baseline."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import yaml
import pytest

from g1_access_push.stage2.s2_01_contract import (
    INVALID_REASONS,
    REQUIRED_AUDIT_FIELDS,
    VALID_FAIL_REASONS,
    ballast_inertia,
    classify_evidence,
    load_config,
    placement_from_robot_max_x,
    resolved_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/stage2/s2_01_box_stand_sanity.yaml"
RUNNER = ROOT / "scripts/stage2_isaac/run_s2_01_box_stand_sanity.py"
EVALUATOR = ROOT / "scripts/stage2/evaluate_s2_01_box_stand_sanity.py"
ENV_SOURCE = ROOT / "src/g1_access_push/sim/stage2/s2_01_box_env.py"


def cfg() -> dict:
    return load_config(CONFIG_PATH)


def test_development_scope_and_prior_invalid_are_not_reused_as_current_reason() -> None:
    baseline = cfg()["baseline"]
    assert baseline == {
        "baseline_id": "S2_LIGHT_BOX_DEVELOPMENT_BASELINE_V1",
        "source": "RESEARCH_LEAD_EXPLICIT_DEVELOPMENT_DECISION_2026_07_27",
        "scope": "SIMULATION_DEVELOPMENT_ONLY",
        "not_claimed_as": [
            "real hardware material measurement", "final Stage 2 randomized range",
            "safety-certified payload", "proof of G1 pushing capacity", "paper experimental result",
        ],
    }
    current_reasons = set(VALID_FAIL_REASONS) | set(INVALID_REASONS)
    assert "BOX_PHYSICS_PARAMETER_SOURCE_AMBIGUOUS" not in current_reasons
    assert "INITIAL_CLEARANCE_CONTRACT_UNRESOLVED" not in current_reasons
    assert "ROBOT_RUNTIME_ENVELOPE_QUERY_FAILED" not in current_reasons
    assert "INITIAL_CLEARANCE_MISMATCH" not in current_reasons


def test_geometry_mass_and_explicit_low_com_contract() -> None:
    config = cfg()
    assert config["geometry"]["size_xyz_m"] == [1.2, 0.6, 1.2]
    assert config["geometry"]["half_extent_xyz_m"] == [0.6, 0.3, 0.6]
    assert config["geometry"]["rear_face_xO_m"] == -0.6
    props = config["mass_properties"]
    assert props["mass_kg"] == 5.0
    assert props["center_of_mass_local_xyz_m"] == [0.0, 0.0, -0.4]
    assert props["com_height_above_bottom_m"] == 0.2 < 0.6
    assert props["authoring_api"] == "UsdPhysics.MassAPI"


def test_ballast_inertia_is_recomputed_from_formula() -> None:
    props = cfg()["mass_properties"]
    computed = ballast_inertia(props["mass_kg"], props["ballast_equivalent_size_xyz_m"])
    expected = props["diagonal_inertia_kg_m2"]
    assert computed == pytest.approx(expected, abs=1.0e-15)
    assert computed[0] == pytest.approx(5.0 / 12.0 * (0.5**2 + 0.2**2))
    assert computed[1] == pytest.approx(5.0 / 12.0 * (1.0**2 + 0.2**2))
    assert computed[2] == pytest.approx(5.0 / 12.0 * (1.0**2 + 0.5**2))
    assert props["principal_axes_quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]


def test_box_ground_material_pair_and_rigid_contact() -> None:
    config = cfg()
    expected = {
        "static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.0,
        "friction_combine_mode": "average", "restitution_combine_mode": "average",
    }
    assert config["materials"]["box"] == expected
    assert config["materials"]["ground"] == expected
    assert config["materials"]["effective_pair"] == expected
    assert config["materials"]["palm_box"] == {"status": "UNRESOLVED", "blocking_s2_01": False}
    assert config["contact_model"]["type"] == "RIGID"
    assert config["contact_model"]["stiffness"]["status"] == "NOT_APPLICABLE"
    assert config["contact_model"]["damping"]["status"] == "NOT_APPLICABLE"


def test_fixed_isolation_placement_and_s2_02_deferrals() -> None:
    placement = cfg()["placement"]
    assert placement["source"] == "RESEARCH_LEAD_S2_01_ISOLATION_PLACEMENT"
    assert placement["box_spawn_center_xyz_m"] == [3.0, 0.0, 0.602]
    assert placement["box_rear_face_x_world_m"] == 2.4
    assert placement["initial_yaw_rad"] == 0.0
    assert placement["robot_collider_envelope"] == {
        "status": "DEFERRED_TO_S2_02", "blocking_s2_01": False,
    }
    for name in ("nominal_base_to_box_distance", "precontact_gap"):
        assert placement[name] == {
            "status": "UNRESOLVED", "blocking_s2_01": False, "resolution_stage": "S2_02",
        }
    source = ENV_SOURCE.read_text(encoding="utf-8")
    assert "pos=(3.0, 0.0, 0.602)" in source


def test_settle_stillness_and_frame_contract() -> None:
    config = cfg()
    assert config["settling"]["control_frames"] == 50
    assert config["settling"]["reference_trace_frame"] == 49
    assert config["evaluation"]["first_frame"] == 50
    assert config["evaluation"]["last_frame"] == 2999
    assert config["evaluation"]["expected_frames"] == 3000
    assert config["evaluation"]["stillness_thresholds"] == {
        "max_box_linear_speed_mps": 0.005,
        "max_box_angular_speed_radps": 0.008726646259971648,
        "max_box_translation_from_reference_m": 0.005,
        "max_box_yaw_change_rad": 0.008726646259971648,
        "max_box_abs_roll_rad": 0.008726646259971648,
        "max_box_abs_pitch_rad": 0.008726646259971648,
        "max_box_vertical_drift_m": 0.003,
    }


def test_stage1_controller_contract_and_structured_checkpoint_prohibition() -> None:
    robot = cfg()["robot"]
    assert robot["controller_checkpoint_sha256"] == "a0151975a757a33f0f5ed236d5616e2a98643abe2b36e50ac9ad01b6dd1f6d7e"
    assert robot["lower_body_command"] == [0.0, 0.0, 0.0, 0.7]
    assert robot["lower_body_action_scale"] == "G1_W_HANDS_AGILE_ACTION_SCALE"
    assert robot["upper_body_action"] == "ZERO_DELTA_DEFAULT_ARMS_AND_WAIST"
    text = RUNNER.read_text(encoding="utf-8") + ENV_SOURCE.read_text(encoding="utf-8")
    assert "G1Stage1NoBoxRecurrentEnvCfg" in text
    assert robot["forbidden_checkpoint_sha256s"] == ["c3e147f90400598fdd97f61022acf599305254a2ccc3fdba1d3ace512ce133a0"]
    assert robot["forbidden_checkpoint_basenames"] == ["model_1999.pt"]
    assert "forbidden_checkpoint_substring" not in text
    assert "OnPolicyRunner" not in text
    assert "train(" not in text


def test_no_attach_precontact_push_or_planner() -> None:
    config = cfg()
    assert config["prohibitions"] == {
        "attach": False, "precontact": False, "pushing": False,
        "planner": False, "training": False, "auto_reset": False,
        "model_1999": False,
    }
    source = (RUNNER.read_text(encoding="utf-8") + ENV_SOURCE.read_text(encoding="utf-8")).lower()
    for forbidden in ("attachstate", "precontactstate", "planner.solve", "push_action"):
        assert forbidden not in source


def test_fail_invalid_are_disjoint_and_valid_negative_is_fail() -> None:
    assert set(VALID_FAIL_REASONS).isdisjoint(INVALID_REASONS)
    config = cfg()
    audits = valid_audits()
    records = valid_records(config)
    records[51]["box_linear_speed_mps"] = 0.006
    result = classify_evidence(config, records, audits, runner_rc=0, final_image_present=True, multiple_isaac_processes=False)
    assert result["status"] == "FAIL"
    assert result["primary_reason"] == "OBJECT_LINEAR_SPEED_EXCEEDED"
    missing = classify_evidence(config, [], {}, runner_rc=1, final_image_present=False, multiple_isaac_processes=False)
    assert missing["status"] == "INVALID"


def test_contact_sensor_initialization_failure_is_invalid() -> None:
    config = cfg()
    audits = valid_audits()
    audits["contact_sensor_audit.json"]["sensor_audit_pass"] = False
    result = classify_evidence(
        config, valid_records(config), audits, runner_rc=0,
        final_image_present=True, multiple_isaac_processes=False,
    )
    assert result["status"] == "INVALID"
    assert result["primary_reason"] == "CONTACT_SENSOR_INITIALIZATION_FAILED"


def test_initial_overlap_and_initial_contact_are_valid_physical_failures() -> None:
    config = cfg()
    overlap_audits = valid_audits()
    overlap_audits["scene_geometry_audit.json"]["overlap_count"] = 1
    overlap_audits["scene_geometry_audit.json"]["scene_query_robot_hit"] = True
    overlap = classify_evidence(
        config, valid_records(config), overlap_audits, runner_rc=0,
        final_image_present=True, multiple_isaac_processes=False,
    )
    assert overlap["status"] == "FAIL"
    assert overlap["primary_reason"] == "INITIAL_ROBOT_BOX_OVERLAP"
    contact_audits = valid_audits()
    contact_audits["contact_sensor_audit.json"]["initial_robot_contact_force_max_n"] = 1.0
    contact = classify_evidence(
        config, valid_records(config), contact_audits, runner_rc=0,
        final_image_present=True, multiple_isaac_processes=False,
    )
    assert contact["status"] == "FAIL"
    assert contact["primary_reason"] == "UNEXPECTED_ROBOT_BOX_CONTACT"


@pytest.mark.parametrize(
    "missing_audit",
    ["box_mass_properties_audit.json", "physics_material_audit.json"],
)
def test_missing_required_mass_or_material_audit_is_invalid(missing_audit: str) -> None:
    config = cfg()
    audits = valid_audits()
    del audits[missing_audit]
    result = classify_evidence(
        config, valid_records(config), audits, runner_rc=0,
        final_image_present=True, multiple_isaac_processes=False,
    )
    assert result["status"] == "INVALID"
    assert result["primary_reason"] == "MISSING_REQUIRED_FIELD"


def test_optional_aabb_unavailable_does_not_block_s2_01() -> None:
    config = cfg()
    audits = valid_audits()
    audits["scene_geometry_audit.json"]["robot_root_subtree_aabb"] = "OPTIONAL_METRIC_UNAVAILABLE"
    result = classify_evidence(
        config, valid_records(config), audits, runner_rc=0,
        final_image_present=True, multiple_isaac_processes=False,
    )
    assert result["status"] == "PASS"


def test_required_runtime_audit_fields_are_frozen() -> None:
    assert set(REQUIRED_AUDIT_FIELDS) == {
        "box_mass_properties_audit.json", "physics_material_audit.json", "scene_geometry_audit.json", "box_rigid_body_audit.json", "contact_sensor_audit.json"
    }
    assert all(fields for fields in REQUIRED_AUDIT_FIELDS.values())


def test_evaluator_is_pure_and_never_uses_bool_dones() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith(("isaaclab", "isaacsim", "omni", "pxr", "carb")) for name in imports)
    assert "bool(dones)" not in source
    assert "bool(done" not in source


def test_runner_launches_app_before_runtime_imports() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert source.index("simulation_app = AppLauncher(args).app") < source.index("import torch")
    ast.parse(source)
    ast.parse(ENV_SOURCE.read_text(encoding="utf-8"))


def test_resolved_config_is_runnable_and_deterministic() -> None:
    first = resolved_config(CONFIG_PATH)
    second = resolved_config(CONFIG_PATH)
    assert first == second
    assert first["runnable"] is True
    assert first["qualification_state"] == "READY_FOR_SINGLE_FORMAL_RUN"
    assert len(first["contract_digest_sha256"]) == 64


def valid_audits() -> dict[str, dict]:
    mass = {key: None for key in REQUIRED_AUDIT_FIELDS["box_mass_properties_audit.json"]}
    mass.update({"tolerance_checks": {"mass": True, "com": True, "inertia": True, "principal_axes": True}, "low_com_pass": True})
    material = {key: None for key in REQUIRED_AUDIT_FIELDS["physics_material_audit.json"]}
    material.update({"pair_pass": True, "box_binding_target": ["box_material"], "ground_binding_target": ["ground_material"]})
    geometry = {key: None for key in REQUIRED_AUDIT_FIELDS["scene_geometry_audit.json"]}
    geometry.update({"overlap_count": 0, "scene_query_robot_hit": False})
    rigid = {key: None for key in REQUIRED_AUDIT_FIELDS["box_rigid_body_audit.json"]}
    rigid.update({"runtime_size_xyz_m": [1.2, 0.6, 1.2], "rigid_body_enabled": True, "kinematic_enabled": False, "gravity_enabled": True})
    sensor = {key: None for key in REQUIRED_AUDIT_FIELDS["contact_sensor_audit.json"]}
    sensor.update({"sensor_audit_pass": True, "initial_robot_contact_force_max_n": 0.0})
    return {
        "box_mass_properties_audit.json": mass,
        "physics_material_audit.json": material,
        "scene_geometry_audit.json": geometry,
        "box_rigid_body_audit.json": rigid,
        "contact_sensor_audit.json": sensor,
    }


def valid_records(config: dict) -> list[dict]:
    return [
        {
            "frame": frame, "finite": True, "robot_box_contact": False,
            "robot_fall": False, "robot_bad_tilt": False,
            "root_height_m": 0.75, "post_initial_reset_count": 0,
            "box_linear_speed_mps": 0.0, "box_angular_speed_radps": 0.0,
            "box_translation_from_reference_m": 0.0, "box_abs_yaw_change_rad": 0.0,
            "box_abs_roll_rad": 0.0, "box_abs_pitch_rad": 0.0,
            "box_abs_vertical_drift_m": 0.0, "box_ground_contact_force_n": 49.05,
        }
        for frame in range(config["evaluation"]["expected_frames"])
    ]


def test_complete_synthetic_evidence_passes() -> None:
    config = cfg()
    result = classify_evidence(
        config, valid_records(config), valid_audits(), runner_rc=0,
        final_image_present=True, multiple_isaac_processes=False,
    )
    assert result["status"] == "PASS"
    assert result["primary_reason"] == "ALL_FROZEN_GATES_PASSED"
