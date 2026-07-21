from __future__ import annotations

from pathlib import Path

import yaml

from g1_access_push.sim.stage1.recurrent_no_box_env_cfg import (
    G1Stage1NoBoxRecurrentEnvCfg,
    RECURRENT_STUDENT_POLICY_PATH,
)
from g1_access_push.sim.stage1.recurrent_student_action import (
    RecurrentStudentLowerBodyAction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/stage1/s1_06_base_motion_hand_hold.yaml"
)


def load_config() -> dict:
    return yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def test_s1_06_uses_recurrent_student() -> None:
    cfg = G1Stage1NoBoxRecurrentEnvCfg()
    lower = cfg.actions.lower_body_joint_pos

    assert cfg.scene.num_envs == 1
    assert lower.class_type is RecurrentStudentLowerBodyAction
    assert Path(lower.policy_path) == RECURRENT_STUDENT_POLICY_PATH


def test_s1_06_commands_are_exact() -> None:
    config = load_config()

    commands = {
        segment["name"]: segment["command"]
        for segment in config["segments"]
    }

    assert commands["forward"] == [0.15, 0.0, 0.0, 0.72]
    assert commands["backward"] == [-0.15, 0.0, 0.0, 0.72]

    assert commands["lateral_left"] == [
        0.0,
        0.20,
        0.0,
        0.72,
    ]
    assert commands["lateral_right"] == [
        0.0,
        -0.20,
        0.0,
        0.72,
    ]

    assert commands["arc_left"] == [
        0.15,
        0.0,
        0.25,
        0.72,
    ]
    assert commands["arc_right"] == [
        0.15,
        0.0,
        -0.25,
        0.72,
    ]


def test_s1_06_evaluation_axes_match_commands() -> None:
    config = load_config()

    axis_index = {
        "x": 0,
        "y": 1,
        "wz": 2,
    }

    for segment in config["segments"]:
        command = segment["command"]

        assert len(command) == 4
        assert command[3] == 0.72
        assert segment["steps"] > 0

        for axis, target in segment.get(
            "evaluate_axes",
            {},
        ).items():
            assert target == command[axis_index[axis]]

    arc_segments = {
        segment["name"]: segment
        for segment in config["segments"]
        if segment.get("require_yaw_displacement")
    }

    assert set(arc_segments) == {
        "arc_left",
        "arc_right",
    }


def test_s1_06_acceptance_thresholds() -> None:
    acceptance = load_config()["acceptance"]

    assert (
        acceptance[
            "maximum_linear_velocity_p95_error_m_s"
        ]
        == 0.12
    )
    assert (
        acceptance[
            "maximum_yaw_rate_p95_error_rad_s"
        ]
        == 0.20
    )
    assert (
        acceptance["minimum_arc_yaw_displacement_rad"]
        == 0.80
    )
    assert (
        acceptance["maximum_hand_position_p95_m"]
        == 0.03
    )
    assert (
        acceptance["maximum_hand_orientation_p95_deg"]
        == 5.0
    )
