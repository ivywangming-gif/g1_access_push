"""Pure regressions for S2-01 config, process, sensor, and evaluator infrastructure."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from g1_access_push.stage2.s2_01_process import (
    build_scene_config_instance,
    derive_effective_runner_status,
    persist_effective_runner_status,
    validate_controller_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stage2/s2_01_box_stand_sanity.yaml"
ENV_SOURCE = ROOT / "src/g1_access_push/sim/stage2/s2_01_box_env.py"
PROCESS_SOURCE = ROOT / "src/g1_access_push/stage2/s2_01_process.py"
RUNNER = ROOT / "scripts/stage2_isaac/run_s2_01_box_stand_sanity.py"
EVALUATOR = ROOT / "scripts/stage2/evaluate_s2_01_box_stand_sanity.py"
LAUNCHER = ROOT / "scripts/stage2_isaac/run_s2_01_corrected_once.sh"
EXPECTED_SHA = "a0151975a757a33f0f5ed236d5616e2a98643abe2b36e50ac9ad01b6dd1f6d7e"
FORBIDDEN_SHA = "c3e147f90400598fdd97f61022acf599305254a2ccc3fdba1d3ace512ce133a0"


class FakeScene:
    def __init__(self) -> None:
        self.terrain = {"material": "stage1"}
        self.robot = {"name": "g1"}
        self.num_envs = 1

    def copy(self):
        return copy.deepcopy(self)


def load_evaluator():
    spec = importlib.util.spec_from_file_location("s2_01_evaluator_test", EVALUATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_complete_runner(run: Path, frames: int = 3000) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "runner_status.json").write_text(json.dumps({
        "status": "COMPLETE", "environment_created": True, "observed_frames": frames,
    }))
    (run / "trace.jsonl").write_text("".join(f'{{"frame": {frame}}}\n' for frame in range(frames)))


def test_ast_rejects_all_stage1_scene_cfg_nested_class_access() -> None:
    for path in (ENV_SOURCE, RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forbidden = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "Stage1SceneCfg"
        ]
        assert forbidden == []


def test_helper_requires_instance_and_builds_independent_copies() -> None:
    with pytest.raises(TypeError, match="SCENE_CONFIG_INSTANCE_REQUIRED"):
        build_scene_config_instance(FakeScene, terrain_cfg={}, additions={})
    original = FakeScene()
    first = build_scene_config_instance(original, terrain_cfg={"material": "s2"}, additions={"box": {"mass": 5.0}})
    second = build_scene_config_instance(original, terrain_cfg={"material": "s2"}, additions={"box": {"mass": 5.0}})
    assert first.terrain and first.robot and first.box and first.num_envs == 1
    assert not hasattr(original, "box")
    first.terrain["material"] = "mutated"
    first.box["mass"] = 9.0
    assert original.terrain == {"material": "stage1"}
    assert second.terrain == {"material": "s2"}
    assert second.box == {"mass": 5.0}


def test_checkpoint_validation_uses_identity_not_whole_config_text(tmp_path: Path) -> None:
    checkpoint = tmp_path / "certified.pt"
    checkpoint.write_bytes(b"checkpoint")
    unrelated_config = {
        "prohibitions": {"model_1999": False},
        "description": "model_1999 is forbidden historical evidence",
    }
    assert "model_1999" in json.dumps(unrelated_config)
    result = validate_controller_checkpoint(checkpoint, EXPECTED_SHA, EXPECTED_SHA, [FORBIDDEN_SHA], ["model_1999.pt"])
    assert result["status"] == "PASS"
    assert result["controller_checkpoint_sha_match"] is True
    assert result["controller_checkpoint_forbidden_sha_match"] is False
    assert result["controller_checkpoint_forbidden_path_match"] is False


def test_checkpoint_forbidden_sha_and_basename_fail(tmp_path: Path) -> None:
    certified = tmp_path / "certified.pt"
    certified.write_bytes(b"checkpoint")
    by_sha = validate_controller_checkpoint(certified, FORBIDDEN_SHA, EXPECTED_SHA, [FORBIDDEN_SHA], ["model_1999.pt"])
    assert by_sha["status"] == "FAIL"
    assert by_sha["reason"] == "FORBIDDEN_CHECKPOINT_SELECTED"
    forbidden_path = tmp_path / "model_1999.pt"
    forbidden_path.write_bytes(b"checkpoint")
    by_name = validate_controller_checkpoint(forbidden_path, EXPECTED_SHA, EXPECTED_SHA, [FORBIDDEN_SHA], ["model_1999.pt"])
    assert by_name["status"] == "FAIL"
    assert by_name["reason"] == "FORBIDDEN_CHECKPOINT_SELECTED"


def test_whole_config_substring_scans_are_absent() -> None:
    source = RUNNER.read_text(encoding="utf-8") + PROCESS_SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        '"model_1999" in json.dumps(config)',
        '"model_1999" in yaml_text',
        "forbidden_substring in whole_config_text",
    ):
        assert forbidden not in source
    assert "forbidden_checkpoint_substring" not in CONFIG.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("marker", "IMPLEMENTATION_EXCEPTION"),
        ("invalid", "IMPLEMENTATION_EXCEPTION"),
        ("missing_trace", "MISSING_TRACE"),
        ("short_trace", "INCOMPLETE_TRACE"),
    ],
)
def test_effective_status_rejects_incomplete_raw_zero(tmp_path: Path, setup: str, reason: str) -> None:
    run = tmp_path / setup
    run.mkdir()
    if setup == "marker":
        (run / "implementation_exception.json").write_text("{}")
    elif setup == "invalid":
        (run / "runner_status.json").write_text(json.dumps({
            "status": "INVALID", "primary_reason": "IMPLEMENTATION_EXCEPTION",
            "environment_created": False, "observed_frames": 0,
        }))
    elif setup == "missing_trace":
        (run / "runner_status.json").write_text(json.dumps({
            "status": "COMPLETE", "environment_created": True, "observed_frames": 3000,
        }))
    else:
        write_complete_runner(run, 2999)
    result = derive_effective_runner_status(run, 0, 3000)
    assert result["runner_raw_rc"] == 0
    assert result["runner_effective_rc"] == 1
    assert result["status"] == "INVALID"
    assert result["primary_reason"] == reason


def test_effective_status_complete_and_raw_seven_persistence(tmp_path: Path) -> None:
    complete = tmp_path / "complete"
    write_complete_runner(complete)
    result = persist_effective_runner_status(complete, 0, 3000)
    assert result["runner_effective_rc"] == 0
    assert result["status"] == "COMPLETE"
    assert (complete / "process_rc/runner_raw.txt").read_text() == "0\n"
    assert (complete / "process_rc/runner_effective.txt").read_text() == "0\n"
    assert (complete / "process_rc/runner.txt").read_text() == "0\n"
    assert json.loads((complete / "runner_effective_status.json").read_text())["status"] == "COMPLETE"

    raw_seven = tmp_path / "raw_seven"
    write_complete_runner(raw_seven)
    result = persist_effective_runner_status(raw_seven, 7, 3000)
    assert result["runner_raw_rc"] == 7
    assert result["runner_effective_rc"] == 7
    assert result["status"] == "INVALID"
    assert (raw_seven / "process_rc/runner_raw.txt").read_text() == "7\n"
    assert (raw_seven / "process_rc/runner.txt").read_text() == "7\n"


def test_process_exception_and_cleanup_preserve_nonzero_rc(tmp_path: Path) -> None:
    marker = tmp_path / "implementation_exception.json"
    cleanup = tmp_path / "cleanup.txt"
    code = "\n".join([
        "from pathlib import Path",
        "from g1_access_push.stage2.s2_01_process import guarded_process_main",
        "def fail(): raise RuntimeError('DELIBERATE_FAILURE')",
        f"def cleanup(): Path({str(cleanup)!r}).write_text('done')",
        f"rc=guarded_process_main(fail, cleanup, exception_path=Path({str(marker)!r}))",
        "raise SystemExit(rc)",
    ])
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}")
    completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "Traceback (most recent call last)" in completed.stderr
    assert cleanup.read_text() == "done"
    assert json.loads(marker.read_text())["primary_reason"] == "IMPLEMENTATION_EXCEPTION"


def test_pipefail_preserves_deliberate_python_rc_7(tmp_path: Path) -> None:
    command = (
        "set -o pipefail; "
        f"{sys.executable} -c 'import sys; sys.exit(7)' | tee {tmp_path / 'tee.log'}; "
        "runner_rc=${PIPESTATUS[0]}; exit ${runner_rc}"
    )
    assert subprocess.run(["bash", "-c", command]).returncode == 7


def test_contact_sensor_source_contract_uses_box_rigid_body() -> None:
    source = ENV_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    sensor_paths = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ContactSensorCfg":
            keyword = next(item for item in node.keywords if item.arg == "prim_path")
            assert isinstance(keyword.value, ast.Constant)
            sensor_paths.append(keyword.value.value)
    assert sensor_paths == ["{ENV_REGEX_NS}/Box", "{ENV_REGEX_NS}/Box"]
    assert not any("/geometry/mesh" in path for path in sensor_paths)
    runner = RUNNER.read_text(encoding="utf-8")
    assert "box_net.data.net_forces_w" in runner
    assert "box_robot.data.force_matrix_w" in runner
    assert "contact_sensor_audit.json" in runner


def test_evaluator_complete_run_ignores_unrelated_traceback_log(tmp_path: Path) -> None:
    evaluator = load_evaluator()
    write_complete_runner(tmp_path)
    persist_effective_runner_status(tmp_path, 0, 3000)
    (tmp_path / "stderr.log").write_text("unrelated diagnostic: Traceback (most recent call last)\n")
    assert evaluator.implementation_exception_evidence(tmp_path) is None


def test_evaluator_marker_priority_and_incomplete_trace(tmp_path: Path) -> None:
    evaluator = load_evaluator()
    marker_run = tmp_path / "marker"
    write_complete_runner(marker_run)
    persist_effective_runner_status(marker_run, 0, 3000)
    (marker_run / "implementation_exception.json").write_text(json.dumps({"exception_message": "boom"}))
    assert evaluator.implementation_exception_evidence(marker_run)["primary_reason"] == "IMPLEMENTATION_EXCEPTION"

    short = tmp_path / "short"
    write_complete_runner(short, 2999)
    persist_effective_runner_status(short, 0, 3000)
    assert evaluator.implementation_exception_evidence(short)["primary_reason"] == "INCOMPLETE_TRACE"


def test_runner_preflight_and_launcher_contract_are_static() -> None:
    source = RUNNER.read_text(encoding="utf-8") + PROCESS_SOURCE.read_text(encoding="utf-8")
    for field in (
        "controller_checkpoint_path", "controller_checkpoint_actual_sha256",
        "controller_checkpoint_expected_sha256", "controller_checkpoint_sha_match",
        "controller_checkpoint_forbidden_sha_match", "controller_checkpoint_forbidden_path_match",
    ):
        assert field in source
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for text in ("runner_raw_rc", "runner_effective_rc", "runner_raw.txt", "runner_effective.txt"):
        assert text in launcher
    assert subprocess.run(["bash", "-n", str(LAUNCHER)]).returncode == 0


def test_pure_helpers_and_evaluator_do_not_import_isaac_or_kit() -> None:
    for path in (PROCESS_SOURCE, EVALUATOR, Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any(name.startswith(("isaaclab", "isaacsim", "omni", "pxr", "carb")) for name in imports)
