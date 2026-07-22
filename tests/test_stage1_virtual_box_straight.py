from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from g1_access_push.sim.stage1.no_box_env_cfg import (
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)
from g1_access_push.sim.stage1.recurrent_no_box_env_cfg import (
    RECURRENT_STUDENT_POLICY_PATH,
    STAGE1_RECURRENT_ACTUATOR_DELAY_STEPS,
    STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES,
)
from g1_access_push.sim.stage1.recurrent_student_action import (
    RecurrentStudentLowerBodyAction,
)
from g1_access_push.sim.stage1.virtual_box_env_cfg import (
    G1Stage1VirtualBoxRecurrentEnvCfg,
)
from g1_access_push.sim.stage1.virtual_box_trajectory import (
    straight_out_hold_back_displacement,
    virtual_surface_contact_targets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/stage1/s1_07a_virtual_box_straight.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_s1_07a_protocol_is_isolated_and_deterministic() -> None:
    config = load_config()

    assert config["test_id"] == ("s1_07a_recurrent_virtual_box_straight")
    assert config["episode_mode"] == "isolated"
    assert config["seed"] == 42
    assert config["warmup_steps"] == 150
    assert config["transient_ignore_steps"] == 150

    trajectory = config["trajectory"]

    assert trajectory["frame"] == "pelvis"
    assert trajectory["axis"] == "x"
    assert trajectory["amplitude_m"] == 0.04
    assert trajectory["hold_steps"] > config["transient_ignore_steps"]
    assert trajectory["settle_steps"] > config["transient_ignore_steps"]


def test_s1_07a_uses_two_disjoint_pose_ik_terms() -> None:
    cfg = G1Stage1VirtualBoxRecurrentEnvCfg()

    left = cfg.actions.left_hand_pose
    right = cfg.actions.right_hand_pose
    lower = cfg.actions.lower_body_joint_pos

    assert left.joint_names == LEFT_ARM_JOINT_NAMES
    assert right.joint_names == RIGHT_ARM_JOINT_NAMES
    assert not set(left.joint_names).intersection(right.joint_names)

    assert left.body_name == "left_hand_palm_link"
    assert right.body_name == "right_hand_palm_link"

    for action in (left, right):
        assert action.controller.command_type == "pose"
        assert action.controller.use_relative_mode is True
        assert action.controller.ik_method == "dls"
        assert action.scale == 1.0

    assert lower.class_type is RecurrentStudentLowerBodyAction
    assert Path(lower.policy_path) == RECURRENT_STUDENT_POLICY_PATH


def test_s1_07a_preserves_deterministic_lower_delay() -> None:
    cfg = G1Stage1VirtualBoxRecurrentEnvCfg()

    assert STAGE1_RECURRENT_ACTUATOR_DELAY_STEPS == 0

    for actuator_name in STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES:
        actuator = cfg.scene.robot.actuators[actuator_name]
        assert actuator.min_delay == 0
        assert actuator.max_delay == 0


def test_straight_profile_has_exact_endpoints() -> None:
    amplitude = 0.04
    move_out = 300
    hold = 300
    move_back = 300
    total = move_out + hold + move_back

    values = [
        straight_out_hold_back_displacement(
            step,
            amplitude_m=amplitude,
            move_out_steps=move_out,
            hold_steps=hold,
            move_back_steps=move_back,
        )
        for step in range(total)
    ]

    assert math.isclose(values[0], 0.0, abs_tol=1.0e-12)
    assert math.isclose(
        values[move_out - 1],
        amplitude,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        values[move_out],
        amplitude,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        values[move_out + hold - 1],
        amplitude,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        values[-1],
        0.0,
        abs_tol=1.0e-12,
    )

    assert all(values[index] <= values[index + 1] + 1.0e-12 for index in range(move_out - 1))

    back_start = move_out + hold

    assert all(
        values[index] >= values[index + 1] - 1.0e-12 for index in range(back_start, total - 1)
    )


def test_virtual_contacts_translate_as_one_rigid_pair() -> None:
    baseline = torch.tensor(
        [
            [0.40, 0.20, 0.10],
            [0.40, -0.20, 0.10],
        ],
        dtype=torch.float64,
    )
    translation = torch.tensor(
        [0.04, 0.0, 0.0],
        dtype=torch.float64,
    )

    target = virtual_surface_contact_targets(
        baseline,
        translation,
    )

    assert torch.allclose(target, baseline + translation)
    assert torch.allclose(
        target[1] - target[0],
        baseline[1] - baseline[0],
    )


def test_s1_07a_thresholds_do_not_relax_prior_limits() -> None:
    acceptance = load_config()["acceptance"]

    assert acceptance["maximum_hand_position_p95_m"] == 0.03
    assert acceptance["maximum_hand_position_max_m"] == 0.06
    assert acceptance["maximum_hand_orientation_p95_deg"] == 5.0
    assert acceptance["maximum_hand_orientation_max_deg"] == 8.0
    assert acceptance["maximum_base_xy_excursion_m"] <= 0.05
    assert acceptance["maximum_arm_torque_ratio"] <= 1.001
    assert "maximum_root_orientation_error_deg" not in acceptance
    assert (
        acceptance["maximum_root_tilt_deg"]
        == load_config()["reference_envelope"]["maximum_root_tilt_deg"]
    )
