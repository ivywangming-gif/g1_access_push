from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from g1_access_push.sim.stage1.virtual_box_env_cfg import (
    G1Stage1VirtualBoxRecurrentEnvCfg,
)
from g1_access_push.sim.stage1.virtual_box_trajectory import (
    derive_equal_budget_small_arc,
    virtual_surface_arc_targets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/stage1/s1_07b_virtual_box_small_arcs.yaml"
STRAIGHT_CONFIG_PATH = PROJECT_ROOT / "configs/stage1/s1_07a_virtual_box_straight.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_s1_07b_protocol_is_symmetric_and_isolated() -> None:
    config = load_config()

    assert config["episode_mode"] == "isolated"
    assert config["seed"] == 42
    assert config["warmup_steps"] == 150
    assert config["transient_ignore_steps"] == 150

    assert config["cases"] == [
        {
            "name": "arc_left",
            "direction": 1,
        },
        {
            "name": "arc_right",
            "direction": -1,
        },
    ]

    allocation = config["trajectory"]["budget_allocation"]

    assert allocation == {
        "midpoint_arc_length_fraction": 0.50,
        "contact_rotation_chord_fraction": 0.50,
    }
    assert math.isclose(
        sum(allocation.values()),
        1.0,
        abs_tol=1.0e-12,
    )


def test_equal_budget_arc_respects_validated_budget() -> None:
    geometry = derive_equal_budget_small_arc(
        contact_separation_m=0.28654807806015015,
        local_motion_budget_m=0.04,
    )

    assert math.isclose(
        geometry["midpoint_arc_length_m"],
        0.02,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        geometry["contact_rotation_chord_m"],
        0.02,
        abs_tol=1.0e-12,
    )
    assert geometry["conservative_contact_displacement_bound_m"] <= 0.04 + 1.0e-12
    assert 0.0 < geometry["peak_yaw_rad"] < math.radians(10.0)


def test_left_and_right_arc_targets_are_mirrored() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.15, -0.12],
            [0.0, -0.15, -0.12],
        ],
        dtype=torch.float64,
    )
    quaternions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    geometry = derive_equal_budget_small_arc(
        contact_separation_m=0.30,
        local_motion_budget_m=0.04,
    )

    left = virtual_surface_arc_targets(
        positions,
        quaternions,
        fraction=1.0,
        direction=1,
        center_radius_m=geometry["center_radius_m"],
        peak_yaw_rad=geometry["peak_yaw_rad"],
    )
    right = virtual_surface_arc_targets(
        positions,
        quaternions,
        fraction=1.0,
        direction=-1,
        center_radius_m=geometry["center_radius_m"],
        peak_yaw_rad=geometry["peak_yaw_rad"],
    )

    left_positions, left_quats, left_translation, left_yaw = left
    right_positions, right_quats, right_translation, right_yaw = right

    assert math.isclose(
        left_yaw,
        -right_yaw,
        abs_tol=1.0e-12,
    )
    assert torch.allclose(
        left_translation[0],
        right_translation[0],
    )
    assert torch.allclose(
        left_translation[1],
        -right_translation[1],
    )
    assert torch.allclose(
        torch.linalg.vector_norm(left_positions[1] - left_positions[0]),
        torch.tensor(0.30, dtype=torch.float64),
    )
    assert torch.allclose(
        torch.linalg.vector_norm(right_positions[1] - right_positions[0]),
        torch.tensor(0.30, dtype=torch.float64),
    )
    assert torch.allclose(
        torch.linalg.vector_norm(left_quats, dim=-1),
        torch.ones(2, dtype=torch.float64),
    )
    assert torch.allclose(
        torch.linalg.vector_norm(right_quats, dim=-1),
        torch.ones(2, dtype=torch.float64),
    )


def test_s1_07b_does_not_relax_s1_07a_limits() -> None:
    arc = load_config()["acceptance"]
    straight = load_config(STRAIGHT_CONFIG_PATH)["acceptance"]

    identical_keys = (
        "maximum_hand_position_p95_m",
        "maximum_hand_position_max_m",
        "maximum_hand_orientation_p95_deg",
        "maximum_hand_orientation_max_deg",
        "maximum_contact_separation_p95_error_m",
        "maximum_contact_separation_max_error_m",
        "maximum_hand_return_p95_m",
        "maximum_hand_return_max_m",
        "maximum_arm_joint_target_p95_error_rad",
        "maximum_base_xy_excursion_m",
        "maximum_root_tilt_deg",
        "minimum_root_height_m",
        "maximum_root_height_m",
        "maximum_arm_torque_ratio",
        "minimum_arm_joint_limit_margin_rad",
    )

    for key in identical_keys:
        assert arc[key] == straight[key]


def test_s1_07b_reuses_verified_action_contract() -> None:
    cfg = G1Stage1VirtualBoxRecurrentEnvCfg()

    assert cfg.actions.left_hand_pose.controller.command_type == "pose"
    assert cfg.actions.left_hand_pose.controller.use_relative_mode is True
    assert cfg.actions.left_hand_pose.controller.ik_method == "dls"
    assert cfg.actions.right_hand_pose.controller.command_type == "pose"
    assert cfg.actions.right_hand_pose.controller.use_relative_mode is True
    assert cfg.actions.right_hand_pose.controller.ik_method == "dls"
