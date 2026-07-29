"""Pure contract checks for S2-03T pelvis-contact path recovery.

These tests intentionally do not import Isaac/Kit.  They pin the previously
certified chest endpoint, the exact arm ordering, the dense static sampling
requirements, and the unchanged formal safety gates.  Runtime evidence is
authoritative only after the isolated campaign has completed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from g1_access_push.stage2.s2_03t_pelvis_contact_path_recovery_contract import (
    ARM_JOINT_NAMES,
    CONTROL_DT_S,
    DEVELOPMENT_CLEARANCE_M,
    DYNAMIC_MARGIN_GATE_RAD,
    FORBIDDEN_FORCE_GATE_N,
    HOLD_STEPS,
    JOINT_RATE_LIMIT_RAD_S,
    LEFT_ARM_JOINT_NAMES,
    MOVE_STEPS,
    REFERENCE_SHA256,
    REFERENCE_STATUS,
    RIGHT_ARM_JOINT_NAMES,
    ROOT_HEIGHT_RANGE_M,
    ROOT_TILT_GATE_DEG,
    SETTLE_STEPS,
    STATIC_MARGIN_GATE_RAD,
    STATIC_SEGMENT_SAMPLES,
    STATIC_STRAIGHT_SAMPLES,
    TORQUE_RATIO_GATE,
    finite,
    mirror_audit,
    minimum_jerk_fraction,
    order_audit,
    path_joint_travel,
    segment_samples,
    three_segment_path,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/stage2/s2_03t_safe_chest_waypoint_path.yaml"


def load_config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_dense_static_sampling_and_trajectory_contract_are_frozen() -> None:
    assert STATIC_STRAIGHT_SAMPLES == 501
    assert STATIC_SEGMENT_SAMPLES == 251
    assert SETTLE_STEPS == 50
    assert MOVE_STEPS == 250
    assert HOLD_STEPS == 500
    assert CONTROL_DT_S == pytest.approx(0.02)
    values = [minimum_jerk_fraction(step, MOVE_STEPS) for step in range(MOVE_STEPS + 1)]
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(left <= right for left, right in zip(values, values[1:]))
    assert values[1] < values[2] < values[-2] < values[-1]

    q0 = [0.0] * 14
    q_clear = [0.1] * 14
    q_lift = [0.2] * 14
    q_chest = [0.3] * 14
    path = three_segment_path(q0, q_clear, q_lift, q_chest)
    assert path == [q0, q_clear, q_lift, q_chest]
    assert len(segment_samples(q0, q_clear)) == STATIC_SEGMENT_SAMPLES
    assert segment_samples(q0, q_clear)[0] == q0
    assert segment_samples(q0, q_clear)[-1] == q_clear
    assert path_joint_travel(path) == pytest.approx(math.sqrt(14.0) * 0.3)


def test_reference_order_and_mirror_mapping_are_explicit() -> None:
    expected = list(ARM_JOINT_NAMES)
    assert expected[0::2] == list(LEFT_ARM_JOINT_NAMES)
    assert expected[1::2] == list(RIGHT_ARM_JOINT_NAMES)
    assert len(expected) == 14
    audit = order_audit(expected, expected, expected)
    assert audit["pass"] is True
    mirror = mirror_audit([0.0] * 7, [0.0] * 7, [0.1, 0.2, 0.3], [0.1, -0.2, 0.3])
    assert mirror["status"] == "PASS"
    assert mirror["palm_y_sum_m"] == pytest.approx(0.0)
    assert mirror_audit([0.0] * 6, [0.0] * 7, [0.0] * 3, [0.0] * 3)["status"] == "INVALID"


def test_finite_contract_fails_closed_for_nan_and_inf() -> None:
    assert finite({"q": [0.0, 1.0], "pose": (2.0,)}) is True
    assert finite({"q": [0.0, float("nan")]}) is False
    assert finite([float("inf")]) is False
    assert finite("METRIC_MISSING") is True


def test_yaml_freezes_reference_endpoint_and_previous_negative_result() -> None:
    config = load_config()
    assert config["schema_version"] == 1
    assert config["stage"] == "S2-03T_PELVIS_CONTACT_PAIR_AND_LEFT_ARM_PATH_RECOVERY"
    assert config["status"] in {
        "WAYPOINT_SEARCH_PENDING",
        "STATIC_COLLISION_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED",
    }
    previous = config["previous_authoritative_result"]
    assert isinstance(previous, dict)
    assert previous["status"] == "FAIL"
    assert previous["primary_reason"] == "SAFETY_GATE_TRIGGERED"
    reference = config["reference"]
    assert isinstance(reference, dict)
    assert reference["sha256"] == REFERENCE_SHA256
    assert reference["target_frame"] == "ROOT"
    assert reference["endpoint_must_remain_unchanged"] is True
    assert reference["arm_joint_order"] == list(ARM_JOINT_NAMES)
    assert len(reference["q_chest_arm_reference_rad"]) == 14
    assert reference["q_chest_arm_reference_rad"] == pytest.approx(
        [
            0.42407718300819397,
            0.3559030294418335,
            -0.061720527708530426,
            0.05856018513441086,
            -0.007883468642830849,
            -0.17393609881401062,
            0.7199919819831848,
            0.8523964881896973,
            0.14376944303512573,
            -0.3120492398738861,
            0.5213500261306763,
            0.4298710525035858,
            -0.002913616131991148,
            0.1546958088874817,
        ]
    )
    assert REFERENCE_STATUS == "KINEMATICALLY_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED"


def test_yaml_preserves_controller_and_formal_gate_values() -> None:
    config = load_config()
    controller = config["controller"]
    assert isinstance(controller, dict)
    assert controller["mode"] == "JOINT_SPACE_MINIMUM_JERK"
    assert controller["control_dt_s"] == pytest.approx(CONTROL_DT_S)
    assert controller["settle_steps"] == SETTLE_STEPS
    assert controller["segment_steps"] == MOVE_STEPS
    assert controller["hold_steps"] == HOLD_STEPS
    assert controller["joint_rate_limit_rad_s"] == pytest.approx(JOINT_RATE_LIMIT_RAD_S)
    assert controller["actor_enabled"] is False
    assert controller["box_used"] is False
    assert controller["walking_enabled"] is False
    assert controller["contact_attach_enabled"] is False
    assert controller["differential_ik_for_formal_move"] is False
    static = config["static_path_audit"]
    assert isinstance(static, dict)
    assert static["straight_path_samples"] == STATIC_STRAIGHT_SAMPLES
    assert static["segment_samples"] == STATIC_SEGMENT_SAMPLES
    assert static["development_clearance_m"] == pytest.approx(DEVELOPMENT_CLEARANCE_M)
    assert static["endpoint_collision_free_is_not_sufficient"] is True

    gates = config["frozen_formal_safety_gates"]
    assert isinstance(gates, dict)
    assert gates["minimum_joint_margin_rad"] == pytest.approx(DYNAMIC_MARGIN_GATE_RAD)
    assert gates["static_joint_margin_rad"] == pytest.approx(STATIC_MARGIN_GATE_RAD)
    assert gates["forbidden_contact_force_n"] == pytest.approx(FORBIDDEN_FORCE_GATE_N)
    assert gates["maximum_torque_ratio"] == pytest.approx(TORQUE_RATIO_GATE)
    assert gates["maximum_root_tilt_deg"] == pytest.approx(ROOT_TILT_GATE_DEG)
    assert gates["root_height_range_m"] == pytest.approx(ROOT_HEIGHT_RANGE_M)


def test_waypoint_search_is_bounded_left_first_and_endpoint_locked() -> None:
    config = load_config()
    search = config["waypoint_search"]
    assert isinstance(search, dict)
    assert search["maximum_segments"] == 3
    assert search["maximum_waypoints"] == 2
    assert search["maximum_candidates"] == 32
    assert search["candidate_generation"] == "BOUNDED_DETERMINISTIC_TEMPLATES"
    assert search["left_arm_first"] is True
    assert search["right_arm_mode"] == "DEFAULT_OR_CERTIFIED_PASS_PATH"
    assert search["endpoint"] == "q_chest_arm_reference_rad"
    assert set(search["search_variables"]) <= set(LEFT_ARM_JOINT_NAMES)
    assert len(search["search_variables"]) >= 4
    selected = search["selected"]
    assert isinstance(selected, dict)
    assert selected["status"] in {"NOT_SELECTED", "STATIC_PASS", "STATIC_COLLISION_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED", "SELECTED", "PASS"}
    if selected["status"] != "NOT_SELECTED":
        assert len(selected["q_clear_arm_rad"]) in {7, 14}
        assert len(selected["q_lift_arm_rad"]) in {7, 14}
        assert selected["static_collision_free"] is True


def test_prohibitions_and_output_contract_are_explicit() -> None:
    config = load_config()
    prohibitions = config["prohibitions"]
    assert isinstance(prohibitions, dict)
    for key in (
        "falcon",
        "ppo",
        "reward_changes",
        "box",
        "walking",
        "contact_attach",
        "planner",
        "global_pelvis_collision_disable",
        "threshold_changes",
        "third_party_changes",
        "active_controller_manifest_update",
    ):
        assert prohibitions[key] is False
    outputs = config["outputs"]
    assert isinstance(outputs, dict)
    assert outputs["pair_audit"] == "reports/stage2/s2_03t_pelvis_contact_pair_audit.json"
    assert set(outputs["videos"]) == {
        "pelvis_collision_debug_front.mp4",
        "pelvis_collision_debug_side.mp4",
        "left_waypoint_front.mp4",
        "left_waypoint_side.mp4",
        "bilateral_waypoint_front.mp4",
        "bilateral_waypoint_side.mp4",
    }
