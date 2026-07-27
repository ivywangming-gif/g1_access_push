"""Pure static tests for S2-00; this file must never import simulator modules."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from g1_access_push.stage2.attach_fsm import AttachState, LEGAL_TRANSITIONS, next_state
from g1_access_push.stage2.contract import (
    FailureReason,
    InvalidReason,
    evaluate_attach_gate,
    unresolved_parameters,
)
from g1_access_push.stage2.resolved_config import resolve_config
from g1_access_push.stage2.result_schema import METRIC_FIELDS, contract_only_result

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stage2/s2_00_attach_light_box_development.yaml"
RISK_CONFIG = ROOT / "configs/stage2/s2_00_risk_prevention.yaml"
PROVENANCE = ROOT / "reports/stage2/s2_00_branch_provenance.json"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_box_geometry_rear_frame_and_frozen_prohibitions() -> None:
    cfg = load_config()
    geometry = cfg["geometry"]
    dimensions = ("box_length_xO", "box_width_yO", "box_height_zO")
    assert [geometry[key]["value"] for key in dimensions] == [1.20, 0.60, 1.20]
    assert geometry["rear_surface_xO"]["value"] == -0.60
    assert geometry["rear_inward_normal_xO"]["value"] == 1.0
    assert geometry["rear_tangent_axis"]["value"] == "yO"
    assert geometry["door_width"]["value"] == 0.80
    assert geometry["door_disabled_in_stage2"]["value"] is True
    assert geometry["doorway_enabled"]["value"] is False
    for key in ("planner_enabled", "pushing_enabled", "isaac_execution_enabled", "runnable"):
        assert cfg[key] is False


def test_mirror_relation_without_fabricating_contact_points() -> None:
    contact = load_config()["contact_geometry"]
    assert contact["mirrored_about_yO_zero"]["value"] is True
    assert contact["equal_contact_height"]["value"] is True
    assert contact["palm_normals_plus_xO"]["value"] is True
    assert contact["tangential_separation"]["value"] == "UNRESOLVED"
    assert contact["contact_height"]["value"] == "UNRESOLVED"


def test_all_required_unresolved_parameters_use_sentinel_and_separate_frequencies() -> None:
    cfg = load_config()
    found = set(unresolved_parameters(cfg))
    required = {
        "object_physics.box_mass",
        "object_physics.box_ground_static_friction",
        "object_physics.box_ground_dynamic_friction",
        "object_physics.palm_box_static_friction",
        "object_physics.palm_box_dynamic_friction",
        "object_physics.restitution",
        "object_physics.contact_stiffness",
        "object_physics.contact_damping",
        "object_physics.box_com_x_offset",
        "object_physics.box_com_y_offset",
        "object_physics.box_com_height",
        "contact_geometry.contact_height",
        "contact_geometry.tangential_separation",
        "contact_geometry.nominal_base_to_box_distance",
        "contact_geometry.precontact_gap",
        "attach_motion.approach_speed",
        "attach_motion.approach_acceleration_limit",
        "attach_motion.approach_jerk_rate_limit",
        "attach_motion.contact_verification_window",
        "attach_motion.attach_hold_duration",
        "attach_gate.left_contact_force_threshold",
        "attach_gate.right_contact_force_threshold",
        "attach_gate.left_contact_impulse_threshold",
        "attach_gate.right_contact_impulse_threshold",
        "attach_gate.excessive_contact_impulse_threshold",
        "attach_gate.contact_force_peak_threshold",
        "attach_gate.contact_force_rate_threshold",
        "attach_gate.object_linear_speed_threshold",
        "attach_gate.object_yaw_rate_threshold",
        "attach_gate.object_translation_drift_threshold",
        "attach_gate.object_yaw_drift_threshold",
        "attach_gate.contact_loss_grace_duration",
        "attach_gate.single_hand_contact_max_duration",
        "robot_stability.joint_limit_safety_margin",
        "robot_stability.torque_limit_ratio_threshold",
        "robot_stability.robot_fall_threshold",
        "robot_stability.robot_bad_tilt_threshold",
        "robot_stability.root_height_acceptable_range",
        "robot_stability.foot_contact_stability_threshold",
        "logging.simulation_control_frequency",
        "logging.contact_safety_supervisor_frequency",
        "logging.logging_trace_frequency",
    }
    assert found == required
    for dotted in found:
        value = cfg
        for key in dotted.split("."):
            value = value[key]
        assert value["status"] == "UNRESOLVED"
        assert value["value"] == "UNRESOLVED"


def test_geometric_center_com_is_rejected_without_authoritative_source(tmp_path: Path) -> None:
    cfg = load_config()
    assert cfg["object_physics"]["box_com_height"]["value"] == "UNRESOLVED"
    cfg["object_physics"]["box_com_height"]["value"] = 0.60
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="geometric-center CoM"):
        resolve_config(bad_path)


def test_fsm_normal_path_failure_edges_and_terminal_states() -> None:
    order = [
        AttachState.RESET,
        AttachState.STAND_SETTLE,
        AttachState.PRECONTACT,
        AttachState.APPROACH_NORMAL,
        AttachState.BILATERAL_CONTACT_VERIFY,
        AttachState.ATTACHED_HOLD,
        AttachState.PASS,
    ]
    for current, requested in zip(order, order[1:]):
        assert next_state(current, requested) is requested
    for current in order[:-1]:
        assert AttachState.FAIL in LEGAL_TRANSITIONS[current]
        assert next_state(
            current, AttachState.FAIL, failure_reason=FailureReason.NONFINITE.value
        ) is AttachState.FAIL
    with pytest.raises(ValueError, match="exact registered failure reason"):
        next_state(AttachState.RESET, AttachState.FAIL)
    with pytest.raises(ValueError):
        next_state(AttachState.APPROACH_NORMAL, AttachState.PASS)
    assert LEGAL_TRANSITIONS[AttachState.PASS] == frozenset()
    assert LEGAL_TRANSITIONS[AttachState.FAIL] == frozenset()


def test_failure_groups_are_disjoint_and_complete() -> None:
    assert len(FailureReason) == 15
    assert len(InvalidReason) == 9
    assert {item.value for item in FailureReason}.isdisjoint(
        {item.value for item in InvalidReason}
    )


def test_attach_gate_missing_unresolved_and_bilateral_requirements() -> None:
    complete = {
        "left_contact": True,
        "right_contact": True,
        "object_linear_speed_mps": 0.0,
        "object_yaw_rate_radps": 0.0,
        "robot_stable": True,
        "no_forbidden_collision": True,
    }
    assert evaluate_attach_gate({"left_contact": True}, {})["status"] == "INVALID_EVIDENCE"
    assert evaluate_attach_gate(complete, {})["status"] == "CONTRACT_UNRESOLVED"
    thresholds = {
        "object_linear_speed_threshold": 0.1,
        "object_yaw_rate_threshold": 0.1,
    }
    single_hand = dict(complete, right_contact=False)
    assert evaluate_attach_gate(single_hand, thresholds)["status"] == "FAIL"
    assert evaluate_attach_gate(complete, thresholds)["status"] == "PASS"


def test_not_run_result_has_complete_metrics_and_no_fake_evidence() -> None:
    result = contract_only_result(
        run_id="s2_00_static",
        config_path="config",
        config_sha256="a" * 64,
        resolved_config_path="resolved",
        code_head="HEAD",
        branch="stage2/attach-light-box",
    )
    assert result["status"] == "NOT_RUN"
    assert result["primary_reason"] == "S2_00_STATIC_CONTRACT_ONLY"
    assert result["evidence"] == {"ATTACH_NOT_RUN": True, "BOX_NOT_PUSHED": True}
    assert result["trace_path"] == "NOT_CREATED"
    assert result["final_image_path"] == "NOT_CREATED"
    assert result["checkpoint"] == "NOT_USED"
    assert set(result["metrics"]) == set(METRIC_FIELDS)
    for key in ("time_out", "fall", "bad_tilt", "nonfinite"):
        assert key in result["metrics"]
    assert "dones" not in result["metrics"]


def test_only_palm_contact_semantics_are_allowed() -> None:
    contact = load_config()["contact_geometry"]
    assert contact["allowed_contact_semantics"]["value"] == [
        "left_palm_allowed_patch",
        "right_palm_allowed_patch",
    ]
    forbidden = contact["forbidden_contact_semantics"]["value"]
    assert {"forearm", "upper_arm", "shoulder", "chest_torso", "pelvis", "thigh", "shank", "foot", "head"}.issubset(forbidden)


def test_wrench_sign_contract() -> None:
    def wrench(fl: float, fr: float, separation: float = 1.0) -> tuple[float, float]:
        return fl + fr, separation * (fr - fl)

    assert wrench(2.0, 2.0) == (4.0, 0.0)
    assert wrench(1.0, 3.0)[1] == -wrench(3.0, 1.0)[1]
    assert wrench(-2.0, 2.0)[0] == 0.0
    assert wrench(-2.0, 2.0)[1] != 0.0


def test_impact_risk_references_config_keys_and_result_metrics() -> None:
    risk = yaml.safe_load(RISK_CONFIG.read_text(encoding="utf-8"))["risks"][0]
    assert risk["risk_id"] == "RISK_A_IMPACT_EJECTION"
    required_refs = {
        "contact_geometry.precontact_gap",
        "attach_motion.approach_speed",
        "attach_motion.approach_acceleration_limit",
        "attach_motion.approach_jerk_rate_limit",
        "attach_gate.left_contact_impulse_threshold",
        "attach_gate.right_contact_impulse_threshold",
        "attach_gate.excessive_contact_impulse_threshold",
        "attach_gate.contact_force_peak_threshold",
        "attach_gate.contact_force_rate_threshold",
    }
    assert set(risk["config_references"]) == required_refs
    required_metrics = {
        "left_contact_impulse_ns",
        "right_contact_impulse_ns",
        "contact_force_peak_n",
        "contact_force_rate_nps",
    }
    assert required_metrics.issubset(risk["required_metrics"])
    assert required_metrics.issubset(METRIC_FIELDS)


def test_resolved_digest_ignores_timestamp_and_source_path(tmp_path: Path) -> None:
    copied = tmp_path / "copied-contract.yaml"
    copied.write_bytes(CONFIG.read_bytes())
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = resolve_config(CONFIG, output_path=first_path)
    second = resolve_config(copied, output_path=second_path)
    assert first["source_config_path"] != second["source_config_path"]
    assert first["contract_digest_sha256"] == second["contract_digest_sha256"]
    assert first["unresolved_count"] == len(first["unresolved_parameters"]) == 42
    assert first["runnable"] is False
    assert first["qualification_state"] == "CONTRACT_ONLY_NOT_EXECUTED"
    for path in (first_path, second_path):
        loaded = json.loads(path.read_text(encoding="utf-8"))
        expected = json.dumps(loaded, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        assert path.read_text(encoding="utf-8") == expected


def test_new_python_files_have_no_forbidden_runtime_imports() -> None:
    forbidden = {"isaacsim", "omni", "carb", "pxr", "isaaclab", "AppLauncher"}
    paths = list((ROOT / "src/g1_access_push/stage2").glob("*.py"))
    paths += list((ROOT / "scripts/stage2").glob("*.py"))
    paths += list((ROOT / "tests/stage2").glob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.names[0].name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        }
        imported |= {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not imported & forbidden, (path, imported & forbidden)


def test_stage1_facts_are_not_rewritten() -> None:
    files = [CONFIG, RISK_CONFIG]
    files += list((ROOT / "src/g1_access_push/stage2").glob("*.py"))
    files += list((ROOT / "scripts/stage2").glob("*.py"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "model_1999" not in text
    assert "active_controller_manifest" not in text
    assert "mean_palm_z" not in text
    assert "scale=1.0" not in text
    assert "G1_W_HANDS_AGILE_ACTION_SCALE" in text


def test_provenance_is_explicitly_a_pre_commit_snapshot() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["snapshot_semantics"] == "PRE_COMMIT_SNAPSHOT"
    assert provenance["commit_created"] is False
    assert provenance["push_performed"] is False
    assert provenance["post_commit_state_intentionally_not_embedded"] is True
    assert provenance["self_sha256_intentionally_omitted"] is True
