"""Static source and process tests for S2-02."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/stage2_isaac/run_s2_02_precontact_audit.py"
LAUNCHER = ROOT / "scripts/stage2_isaac/run_s2_02_once.sh"
ENV = ROOT / "src/g1_access_push/sim/stage2/s2_02_precontact_env.py"
EVALUATOR = ROOT / "scripts/stage2/evaluate_s2_02_precontact_audit.py"


def test_app_launcher_precedes_runtime_imports() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert source.index("simulation_app = AppLauncher(args).app") < source.index("import torch")
    ast.parse(source)
    ast.parse(ENV.read_text(encoding="utf-8"))


def test_runner_uses_proven_differential_ik_and_certified_recurrent_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8") + ENV.read_text(encoding="utf-8")
    assert "G1Stage1VirtualBoxRecurrentEnvCfg" in source
    assert "G1_W_HANDS_AGILE_ACTION_SCALE" in source
    assert '"left_hand_pose": 6' in source
    assert '"right_hand_pose": 6' in source
    assert "Pink" not in source
    assert "OnPolicyRunner" not in source
    assert "train(" not in source


def test_runtime_geometry_and_forbidden_collision_are_measured() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for token in (
        "pelvis_pose_world", "palm_positions_in_pelvis_m", "arm_joint_limits_rad",
        "robot_collision_paths", "box_rear_face_local_x_m", "overlap_query",
        "robot_box_contact_force_n", "candidate_records.json",
    ):
        assert token in source


def test_box_runtime_mass_com_inertia_are_set_and_read_back() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for token in ("get_masses", "set_masses", "get_coms", "set_coms", "get_inertias", "set_inertias"):
        assert token in source
    assert "runtime_mass_properties_audit.json" in source


def test_preflight_and_formal_modes_are_mutually_exclusive() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'mode = parser.add_mutually_exclusive_group(required=True)' in source
    assert 'mode.add_argument("--preflight-only"' in source
    assert 'mode.add_argument("--formal"' in source
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "S2_02_PREFLIGHT_RUN_ROOT" in launcher
    assert "S2_02_PREFLIGHT_CONFIG_SHA_MISMATCH" in launcher


def test_formal_fsm_and_single_episode_contract_are_explicit() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for phase in ("RESET", "STAND_SETTLE", "MOVE_TO_PRECONTACT", "PRECONTACT_HOLD", "FINAL_AUDIT"):
        assert phase in source
    assert "post_initial_reset_count" in source
    assert "expected_frames = settle_steps + move_steps + hold_steps" in source


def test_evaluator_is_pure_and_writes_authoritative_result() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith(("isaaclab", "isaacsim", "omni", "pxr", "carb")) for name in imports)
    assert 'write_json(args.run_root / "result.json"' in source


def test_no_attach_push_planner_or_training_execution() -> None:
    source = (RUNNER.read_text(encoding="utf-8") + ENV.read_text(encoding="utf-8")).lower()
    for forbidden in ("attachstate", "push_action", "planner.solve", "onpolicyrunner"):
        assert forbidden not in source
