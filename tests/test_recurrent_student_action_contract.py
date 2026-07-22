from __future__ import annotations

from pathlib import Path

import isaaclab.utils.string as string_utils
import torch
from agile.rl_env.assets.robots.unitree_g1 import (
    LEG_JOINT_NAMES,
    NO_HAND_JOINT_NAMES,
)

from g1_access_push.sim.stage1.recurrent_no_box_env_cfg import (
    RECURRENT_STUDENT_POLICY_PATH,
    G1Stage1NoBoxRecurrentEnvCfg,
)
from g1_access_push.sim.stage1.recurrent_student_action import (
    STUDENT_INPUT_DIM,
    STUDENT_OBSERVATION_DIM,
    STUDENT_OUTPUT_DIM,
    RecurrentStudentLowerBodyAction,
    compose_recurrent_student_input,
)

# Concrete joint contract used by the exported recurrent student.
#
# NO_HAND_JOINT_NAMES and LEG_JOINT_NAMES are lists of regex
# selectors, not lists containing one entry per resolved joint.
G1_RECURRENT_NO_HAND_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

G1_RECURRENT_LEG_JOINT_NAMES = tuple(
    name
    for name in G1_RECURRENT_NO_HAND_JOINT_NAMES
    if any(
        token in name
        for token in (
            "_hip_",
            "_knee_",
            "_ankle_",
        )
    )
)


def test_student_dimensions() -> None:
    no_hand_ids, no_hand_names = string_utils.resolve_matching_names(
        NO_HAND_JOINT_NAMES,
        list(G1_RECURRENT_NO_HAND_JOINT_NAMES),
    )
    leg_ids, leg_names = string_utils.resolve_matching_names(
        LEG_JOINT_NAMES,
        list(G1_RECURRENT_NO_HAND_JOINT_NAMES),
    )

    # The imported constants contain regex selectors:
    # seven expressions resolve to 29 non-hand joints.
    assert len(NO_HAND_JOINT_NAMES) == 7
    assert len(no_hand_ids) == 29
    assert len(no_hand_names) == 29
    assert set(no_hand_names) == set(G1_RECURRENT_NO_HAND_JOINT_NAMES)

    # Three expressions resolve to the 12 leg joints
    # produced by the recurrent student.
    assert len(LEG_JOINT_NAMES) == 3
    assert len(leg_ids) == STUDENT_OUTPUT_DIM
    assert len(leg_names) == STUDENT_OUTPUT_DIM
    assert set(leg_names) == set(G1_RECURRENT_LEG_JOINT_NAMES)

    assert STUDENT_OUTPUT_DIM == 12
    assert STUDENT_OBSERVATION_DIM == 64
    assert STUDENT_INPUT_DIM == 80


def test_compose_single_environment_input() -> None:
    command = torch.zeros((1, 4))
    observation = torch.zeros((1, 64))
    previous_action = torch.zeros((1, 12))

    result = compose_recurrent_student_input(
        command,
        observation,
        previous_action,
    )

    assert result.shape == (80,)


def test_compose_rejects_batched_input() -> None:
    command = torch.zeros((2, 4))
    observation = torch.zeros((2, 64))
    previous_action = torch.zeros((2, 12))

    try:
        compose_recurrent_student_input(
            command,
            observation,
            previous_action,
        )
    except ValueError as exc:
        assert "exactly one environment" in str(exc)
    else:
        raise AssertionError("Expected batched input rejection")


def test_recurrent_environment_selects_custom_action() -> None:
    cfg = G1Stage1NoBoxRecurrentEnvCfg()

    lower = cfg.actions.lower_body_joint_pos

    assert cfg.scene.num_envs == 1
    assert lower.class_type is RecurrentStudentLowerBodyAction
    assert lower.obs_group_name == "student_policy"
    assert Path(lower.policy_path) == RECURRENT_STUDENT_POLICY_PATH


def test_torchscript_contract_on_cpu() -> None:
    model = torch.jit.load(
        str(RECURRENT_STUDENT_POLICY_PATH),
        map_location="cpu",
    )
    model.eval()

    with torch.no_grad():
        model.hidden_state.zero_()
        model.cell_state.zero_()

    with torch.inference_mode():
        output = model(torch.zeros(80))

    assert output.shape == (12,)
    assert torch.isfinite(output).all()
