"""Static source contracts for the S2-03 runtime slice."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "src/g1_access_push/sim/stage2/s2_03_attach_env.py"
RUNNER = ROOT / "scripts/stage2_isaac/run_s2_03_attach_only.py"
LAUNCHER = ROOT / "scripts/stage2_isaac/run_s2_03_once.sh"
STATUS = ROOT / "scripts/stage2/derive_s2_03_runner_status.py"


def test_two_independent_single_body_palm_sensors_filter_box() -> None:
    source = ENV.read_text(); tree = ast.parse(source)
    assert 'LEFT_PALM_SENSOR_PRIM_PATH = "{ENV_REGEX_NS}/Robot/left_hand/left_hand_palm_link"' in source
    assert 'RIGHT_PALM_SENSOR_PRIM_PATH = "{ENV_REGEX_NS}/Robot/right_hand/right_hand_palm_link"' in source
    assert 'BOX_FILTER_EXPRESSIONS = ("{ENV_REGEX_NS}/Box",)' in source
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ContactSensorCfg"]
    assert len(calls) == 2
    assert "left_palm_box_contact" in source and "right_palm_box_contact" in source


def test_runner_uses_fsm_normal_only_and_records_bilateral_evidence() -> None:
    source = RUNNER.read_text()
    for token in ("AttachState.RESET", "AttachState.STAND_SETTLE", "AttachState.PRECONTACT", "AttachState.APPROACH_NORMAL", "AttachState.BILATERAL_CONTACT_VERIFY", "AttachState.ATTACHED_HOLD"):
        assert token in source
    for metric in ("left_contact", "right_contact", "left_force_n", "right_force_n", "left_impulse_ns", "right_impulse_ns", "contact_force_rate_nps", "forbidden_contact_links"):
        assert metric in source
    assert "advance_rate_limited" in source
    assert "failure_unload_steps" in source
    assert "box_displacement_subgoal" not in source


def test_effective_status_honors_complete_terminal_json_and_rejects_exception(tmp_path: Path) -> None:
    (tmp_path / "runner_status.json").write_text(json.dumps({"status": "COMPLETE", "environment_created": True, "observed_frames": 1}))
    (tmp_path / "trace.jsonl").write_text("{}\n")
    (tmp_path / "fsm_outcome.json").write_text(json.dumps({"terminal_state": "FAIL", "failure_reason": "FORBIDDEN_BODY_BOX_COLLISION"}))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    completed = subprocess.run([sys.executable, str(STATUS), "--run-root", str(tmp_path), "--raw-rc", "1", "--mode", "formal"], env=env, check=False)
    assert completed.returncode == 0
    effective = json.loads((tmp_path / "runner_effective_status.json").read_text())
    assert effective["runner_effective_rc"] == 0
    assert effective["raw_rc_nonzero_after_authoritative_completion"] is True
    (tmp_path / "implementation_exception.json").write_text("{}")
    invalid = subprocess.run([sys.executable, str(STATUS), "--run-root", str(tmp_path), "--raw-rc", "0", "--mode", "formal"], env=env, check=False)
    assert invalid.returncode == 1
    assert json.loads((tmp_path / "runner_effective_status.json").read_text())["primary_reason"] == "IMPLEMENTATION_EXCEPTION"


def test_launcher_gates_formal_on_same_passed_preflight_sha() -> None:
    source = LAUNCHER.read_text()
    assert "S2_03_PREFLIGHT_RUN_ROOT" in source
    assert "S2_03_PREFLIGHT_CONFIG_SHA_MISMATCH" in source
    assert "--preflight-only" in source and "--formal" in source
    assert "process_rc/evaluator.txt" in source
