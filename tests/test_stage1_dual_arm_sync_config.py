from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/stage1/s1_05_dual_arm_sync.yaml"


def test_dual_arm_sync_config_is_conservative() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["test_id"] == "s1_05_dual_arm_sync"
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
    assert acceptance["maximum_peak_time_difference_s"] <= 0.20
    assert acceptance["maximum_arm_torque_ratio"] <= 1.001
