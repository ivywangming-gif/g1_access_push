"""Pure static contract tests for the isolated chest-prepose slice."""

from __future__ import annotations

import hashlib
from pathlib import Path

from g1_access_push.stage2.s2_03t_chest_prepose_contract import (
    ARM_JOINT_NAMES,
    HOLD_SECONDS,
    HOLD_STEPS,
    LEFT_ARM_JOINT_NAMES,
    MOVE_STEPS,
    RIGHT_ARM_JOINT_NAMES,
    SETTLE_STEPS,
    S2_02_CANDIDATE,
    certified_object_local_targets,
    minimum_jerk_fraction,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "src/g1_access_push/sim/stage2/s2_03t_chest_prepose_env_cfg.py"
RUNNER = ROOT / "scripts/stage2_isaac/run_s2_03t_chest_prepose.py"
LAUNCHER = ROOT / "scripts/stage2_isaac/run_s2_03t_chest_prepose_once.sh"


def test_certified_s2_02_target_is_mirrored_and_frozen() -> None:
    assert S2_02_CANDIDATE["precontact_gap_m"] == 0.06
    assert S2_02_CANDIDATE["contact_height_m"] == 0.62
    assert S2_02_CANDIDATE["tangential_separation_m"] == 0.30
    assert certified_object_local_targets() == [
        [-1.1002388986945153, 0.15, 0.020000000000000018],
        [-1.1002388986945153, -0.15, 0.020000000000000018],
    ]


def test_minimum_jerk_is_monotonic_and_has_zero_endpoint_velocity() -> None:
    values = [minimum_jerk_fraction(step, MOVE_STEPS) for step in range(MOVE_STEPS + 1)]
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(left <= right for left, right in zip(values, values[1:]))
    assert HOLD_SECONDS == HOLD_STEPS * 0.02


def test_interleaved_arm_order_and_side_slices_are_frozen() -> None:
    assert ARM_JOINT_NAMES[0::2] == LEFT_ARM_JOINT_NAMES
    assert ARM_JOINT_NAMES[1::2] == RIGHT_ARM_JOINT_NAMES
    assert len(ARM_JOINT_NAMES) == 14


def test_no_box_runner_uses_s2_02_ik_and_frozen_lower_body() -> None:
    config = CONFIG.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "Stage1SceneCfg" in config
    assert "G1_W_HANDS_AGILE_ACTION_SCALE" in config
    assert "DifferentialInverseKinematicsActionCfg" in config
    assert "RigidObjectCfg" not in config
    assert "S203TContactEnvCfg" not in config
    assert "DifferentialIK" in runner
    assert "minimum_jerk_fraction" in runner
    assert "no_box" in runner
    assert "S203TContactEnv" not in runner
    assert "OnPolicyRunner" not in runner
    assert "PPO" not in runner


def test_launcher_is_single_run_root_and_camera_only() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert ': "${RUN_ROOT:?RUN_ROOT must be provided}"' in launcher
    assert 'if [[ -e "${RUN_ROOT}" ]]' in launcher
    assert "--headless --enable_cameras --device cuda:0" in launcher
    assert "run_s2_03t_chest_prepose.py" in launcher
    assert "PPO" not in launcher
    assert "train" not in launcher.lower()


def test_runner_uses_manager_based_env_two_tuple_step() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert "observation, extras = env.step(actions)" in runner
    assert "terminated" not in runner
    assert "truncated" not in runner


def test_reference_sha_contract_is_not_retyped() -> None:
    contract = (
        ROOT / "src/g1_access_push/stage2/s2_03t_chest_prepose_contract.py"
    ).read_text(encoding="utf-8")
    assert "REFERENCE_SHA256 =" in contract
    digest = hashlib.sha256(
        Path(
            "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
            "s2_03t_contract_smoke_20260728_104740/precontact_reference.pt"
        ).read_bytes()
    ).hexdigest()
    assert digest == "1e537e075af7f888fb95a63501d3976d0d7740eff51c6bd9000f574b00cd0e6c"
