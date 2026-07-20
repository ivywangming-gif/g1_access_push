from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "test_id", "active_side"),
    [
        (
            "s1_03_left_arm_motion.yaml",
            "s1_03_left_arm_motion",
            "left",
        ),
        (
            "s1_04_right_arm_motion.yaml",
            "s1_04_right_arm_motion",
            "right",
        ),
    ],
)
def test_single_arm_motion_config_is_conservative(
    filename: str,
    test_id: str,
    active_side: str,
) -> None:
    path = PROJECT_ROOT / "configs" / "stage1" / filename
    config = yaml.safe_load(path.read_text())

    assert config["test_id"] == test_id
    assert config["active_side"] == active_side
    assert config["warmup_steps"] >= 100
    assert config["motion_steps"] >= 500
    assert config["settle_steps"] >= 100

    assert config["base_command"] == {
        "lin_vel_x": 0.0,
        "lin_vel_y": 0.0,
        "ang_vel_z": 0.0,
        "base_height": 0.72,
    }

    offsets = config["joint_offsets_rad"]
    assert 0.0 < offsets["shoulder_pitch_joint"] <= 0.20
    assert 0.0 < offsets["elbow_joint"] <= 0.25

    acceptance = config["acceptance"]
    assert acceptance["minimum_active_hand_peak_displacement_m"] >= 0.03
    assert acceptance["maximum_arm_torque_ratio"] <= 1.001
