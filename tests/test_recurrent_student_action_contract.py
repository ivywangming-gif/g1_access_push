from __future__ import annotations

from pathlib import Path

import torch

from agile.rl_env.assets.robots.unitree_g1 import (
    LEG_JOINT_NAMES,
    NO_HAND_JOINT_NAMES,
)
from g1_access_push.sim.stage1.recurrent_no_box_env_cfg import (
    G1Stage1NoBoxRecurrentEnvCfg,
    RECURRENT_STUDENT_POLICY_PATH,
)
from g1_access_push.sim.stage1.recurrent_student_action import (
    STUDENT_INPUT_DIM,
    STUDENT_OBSERVATION_DIM,
    STUDENT_OUTPUT_DIM,
    RecurrentStudentLowerBodyAction,
    compose_recurrent_student_input,
)


def test_student_dimensions() -> None:
    assert len(NO_HAND_JOINT_NAMES) == 29
    assert len(LEG_JOINT_NAMES) == STUDENT_OUTPUT_DIM
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
