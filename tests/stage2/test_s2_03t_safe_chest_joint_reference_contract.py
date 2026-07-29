from __future__ import annotations

import json
from pathlib import Path

from g1_access_push.stage2.s2_03t_safe_chest_joint_reference_contract import (
    ARM_JOINT_NAMES,
    JOINT_MARGIN_GATE_RAD,
    MOVE_STEPS,
    OLD_GEOMETRY_AUDIT_PATH,
    OLD_TRACE_PATH,
    STATIC_MARGIN_GATE_RAD,
    audit_old_trace,
    certified_object_local_targets,
    minimum_jerk_fraction,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/stage2_isaac/run_s2_03t_safe_chest_joint_reference.py"
CONFIG = ROOT / "configs/stage2/s2_03t_safe_chest_joint_reference.yaml"
ENV_CFG = ROOT / "src/g1_access_push/sim/stage2/s2_03t_safe_chest_joint_reference_env_cfg.py"


def test_old_trace_recomputation_freezes_failure() -> None:
    trace = json.loads(OLD_TRACE_PATH.read_text(encoding="utf-8"))
    geometry = json.loads(OLD_GEOMETRY_AUDIT_PATH.read_text(encoding="utf-8"))
    audit = audit_old_trace(trace, geometry)
    assert audit["limiting_joint"] == "right_elbow_joint"
    assert audit["target_margin_rad"] < JOINT_MARGIN_GATE_RAD
    assert audit["actual_margin_rad"] < JOINT_MARGIN_GATE_RAD
    assert audit["old_target_can_pass_0p10_margin_gate"] == "NO"
    assert audit["pre_formal_move_target_margin_below_0p10"] is False
    assert audit["max_actual_delta_q_rad"] > 0.05
    assert audit["max_target_delta_q_rad"] > 0.05


def test_minimum_jerk_has_zero_endpoint_velocity() -> None:
    values = [minimum_jerk_fraction(step, MOVE_STEPS) for step in range(MOVE_STEPS + 1)]
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(a <= b for a, b in zip(values, values[1:]))
    assert values[1] < values[2] < values[-2] < values[-1]


def test_frozen_geometry_and_joint_contract() -> None:
    targets = certified_object_local_targets()
    assert targets[0][0] == targets[1][0]
    assert targets[0][1] == -targets[1][1]
    assert targets[0][2] == targets[1][2]
    assert len(ARM_JOINT_NAMES) == 14
    assert STATIC_MARGIN_GATE_RAD == 0.12
    assert "DifferentialInverseKinematicsActionCfg" not in ENV_CFG.read_text(encoding="utf-8")
    source = RUNNER.read_text(encoding="utf-8")
    assert "JOINT_SPACE_MINIMUM_JERK" in source
    assert "differential_ik_used_for_formal_move" in source
    assert "FORMAL_JOINT_SPACE_MOVE_HOLD" in source


def test_yaml_records_frozen_status_and_prohibitions() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "KINEMATICALLY_VALIDATED_NOT_YET_DYNAMICALLY_CERTIFIED" in text
    assert "minimum_static_joint_margin_rad: 0.12" in text
    assert "differential_ik_used_for_formal_move: false" in text
    assert "box_used: false" in text
    assert "ppo_started: false" in text
