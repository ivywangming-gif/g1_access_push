from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "configs" / "stage1" / "wbc_baseline.yaml"
ZERO_HOLD_PATH = PROJECT_ROOT / "configs" / "stage1" / "s1_00a_official_zero_hold.yaml"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "stage1" / "run_s1_00a_official_zero_hold.sh"


def test_stage1_wbc_baseline_manifest() -> None:
    manifest = yaml.safe_load(BASELINE_PATH.read_text())

    assert manifest["stage"] == 1
    assert manifest["task"]["id"] == "Velocity-Height-G1-v0"
    assert manifest["agile"]["tag"] == "v1.2"
    assert manifest["isaac_lab"]["tag"] == "v2.3.1"

    checkpoint = manifest["checkpoint"]
    assert checkpoint["format"] == "torchscript"
    assert len(checkpoint["sha256"]) == 64

    evaluation = manifest["zero_hold_evaluation"]
    assert evaluation["episode_length_s"] == 60.0
    assert evaluation["num_envs"] == 1
    assert evaluation["nominal_base_height_m"] == 0.72

    acceptance = manifest["acceptance"]
    assert acceptance["minimum_frames"] >= 2950
    assert acceptance["maximum_final_xy_drift_m"] <= 0.25


def test_stage1_zero_hold_schedule_is_stationary() -> None:
    config = yaml.safe_load(ZERO_HOLD_PATH.read_text())["evaluation"]

    assert config["task_name"] == "Velocity-Height-G1-v0"
    assert config["num_envs"] == 1
    assert config["num_episodes"] == 1
    assert config["episode_length_s"] == 60.0
    assert config["env_overrides"]["events"]["disable_all"] is True

    environments = config["environments"]
    assert len(environments) == 1
    assert environments[0]["env_ids"] == [0]

    schedule = environments[0]["schedule"]
    assert len(schedule) == 1
    assert schedule[0]["time"] == 0.0

    command = schedule[0]["commands"]["base_velocity"]
    assert command == {
        "lin_vel_x": 0.0,
        "lin_vel_y": 0.0,
        "ang_vel_z": 0.0,
        "base_height": 0.72,
    }


def test_stage1_zero_hold_runner_exists_and_is_executable() -> None:
    assert RUNNER_PATH.is_file()
    assert RUNNER_PATH.stat().st_mode & 0o100
