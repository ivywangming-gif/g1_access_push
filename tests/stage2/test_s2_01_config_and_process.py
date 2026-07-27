"""Pure regressions for S2-01 config construction and process status handling."""

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

from g1_access_push.stage2.s2_01_process import build_scene_config_instance

ROOT = Path(__file__).resolve().parents[2]
ENV_SOURCE = ROOT / "src/g1_access_push/sim/stage2/s2_01_box_env.py"
RUNNER = ROOT / "scripts/stage2_isaac/run_s2_01_box_stand_sanity.py"
EVALUATOR = ROOT / "scripts/stage2/evaluate_s2_01_box_stand_sanity.py"
LAUNCHER = ROOT / "scripts/stage2_isaac/run_s2_01_corrected_once.sh"


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
    assert original.terrain == {"material": "stage1"}
    first.terrain["material"] = "mutated"
    first.box["mass"] = 9.0
    assert second.terrain == {"material": "s2"}
    assert second.box == {"mass": 5.0}


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
    payload = json.loads(marker.read_text())
    assert payload["primary_reason"] == "IMPLEMENTATION_EXCEPTION"
    assert payload["exception_type"] == "RuntimeError"


def test_pipefail_preserves_deliberate_python_rc_7(tmp_path: Path) -> None:
    command = (
        "set -o pipefail; "
        f"{sys.executable} -c 'import sys; sys.exit(7)' | tee {tmp_path / 'tee.log'}; "
        "runner_rc=${PIPESTATUS[0]}; exit ${runner_rc}"
    )
    assert subprocess.run(["bash", "-c", command]).returncode == 7


def test_evaluator_exception_priority_and_plain_missing_trace(tmp_path: Path) -> None:
    evaluator = load_evaluator()
    traceback_run = tmp_path / "traceback"
    traceback_run.mkdir()
    (traceback_run / "stderr.log").write_text("Traceback (most recent call last)\nboom\n")
    (traceback_run / "process_rc").mkdir()
    (traceback_run / "process_rc/runner.txt").write_text("0\n")
    assert evaluator.implementation_exception_evidence(traceback_run)["source"] == "DERIVED_PROCESS_EVIDENCE"

    marker_run = tmp_path / "marker"
    marker_run.mkdir()
    (marker_run / "implementation_exception.json").write_text(json.dumps({"primary_reason": "IMPLEMENTATION_EXCEPTION"}))
    (marker_run / "process_rc").mkdir()
    (marker_run / "process_rc/runner.txt").write_text("1\n")
    assert evaluator.implementation_exception_evidence(marker_run)["source"] == "implementation_exception.json"

    empty_run = tmp_path / "empty"
    empty_run.mkdir()
    assert evaluator.implementation_exception_evidence(empty_run) is None
    complete_run = tmp_path / "complete"
    complete_run.mkdir()
    (complete_run / "trace.jsonl").write_text("{}\n")
    (complete_run / "runner_status.json").write_text(json.dumps({"environment_created": True, "observed_frames": 1}))
    assert evaluator.implementation_exception_evidence(complete_run) is None


def test_runner_preflight_and_launcher_contract_are_static() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for field in (
        "CONFIG_INSTANCE_PREFLIGHT_FAILED", "config_is_instance", "terrain_present", "robot_present",
        "box_present", "num_envs", "doorway_enabled", "nearby_obstacles_enabled",
        "stage1_contract_sha", "resolved_config_sha", "failure_reason",
    ):
        assert field in source
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "set -eo pipefail" in launcher
    assert "runner_rc=$?" in launcher
    assert "stdout.log" in launcher and "stderr.log" in launcher
    assert subprocess.run(["bash", "-n", str(LAUNCHER)]).returncode == 0


def test_pure_helper_and_tests_do_not_import_isaac_or_kit() -> None:
    helper = ROOT / "src/g1_access_push/stage2/s2_01_process.py"
    for path in (helper, Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any(name.startswith(("isaaclab", "isaacsim", "omni", "pxr", "carb")) for name in imports)
