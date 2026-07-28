"""Pure/static contract tests for the crash-resilient S2-03T campaign."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/stage2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import finalize_s2_03t_redesign_campaign as finalizer  # noqa: E402
import s2_03t_redesign_campaign as campaign  # noqa: E402

SHELL = SCRIPTS / "run_s2_03t_redesign_campaign.sh"
DRIVER = SCRIPTS / "s2_03t_redesign_campaign.py"
FINALIZER = SCRIPTS / "finalize_s2_03t_redesign_campaign.py"


def pilot_result(**metric_overrides: float) -> dict:
    base = {
        "bilateral_contact_fraction": 0.20,
        "verify_fraction": 0.08,
        "hard_safety_violation_fraction": 0.0,
        "pushing_fraction": 0.0,
        "fall_fraction": 0.0,
        "nonfinite_fraction": 0.0,
        "action_saturation_fraction": 0.10,
    }
    base.update(metric_overrides)
    return {
        "status": "PASS",
        "finite": True,
        "completed_iterations": 200,
        "checkpoints": [{"path": "/outside/model_199.pt", "sha256": "a" * 64}],
        "curve": [{**base, "iteration": index} for index in range(200)],
    }


def test_launcher_is_strict_self_contained_and_single_instance() -> None:
    subprocess.run(("bash", "-n", str(SHELL)), cwd=ROOT, check=True)
    source = SHELL.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in source
    assert "source /root/autodl-tmp/robotics/autodl_env.sh" in source
    assert "conda activate /root/autodl-tmp/conda/envs/agile_env" in source
    assert "flock -n 9" in source
    assert "RUN_ROOT_ALREADY_EXISTS" in source
    assert "campaign.console.log" in source
    assert "trap shell_exit_trap EXIT ERR" in source
    assert "trap 'exit 130' INT" in source
    assert "trap 'exit 143' TERM" in source
    assert "git add ." not in source and "git add -A" not in source


def test_atomic_state_and_history_round_trip(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    history = tmp_path / "stage_history.jsonl"
    campaign.atomic_write_json(status, {"status": "RUNNING", "value": 1.25})
    campaign.append_jsonl(history, {"stage": "A", "rc": 0})
    campaign.append_jsonl(history, {"stage": "B", "rc": 2})
    assert json.loads(status.read_text(encoding="utf-8"))["value"] == 1.25
    assert [
        json.loads(line)["stage"] for line in history.read_text(encoding="utf-8").splitlines()
    ] == ["A", "B"]
    assert not list(tmp_path.glob("*.tmp*"))
    source = DRIVER.read_text(encoding="utf-8")
    assert "os.fsync" in source and "os.replace" in source
    assert campaign.HEARTBEAT_INTERVAL_SECONDS <= 60


def test_shell_exit_preserves_terminal_invalid_reason(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    campaign.atomic_write_json(
        status_path,
        {
            "campaign_status": "INVALID",
            "stage": "DONE",
            "primary_reason": "PRECISE_INVALID_REASON",
        },
    )
    rc = campaign.record_shell_exit(
        SimpleNamespace(run_root=tmp_path, exit_code=2, command="CAMPAIGN_DRIVER")
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert status["campaign_status"] == "INVALID"
    assert status["primary_reason"] == "PRECISE_INVALID_REASON"
    assert status["shell_exit"]["reason"] == "SHELL_EXIT_AFTER_TERMINAL_STATUS"


def test_pilot_gate_is_one_shot_strict_and_finite() -> None:
    passed = campaign.evaluate_pilot_gate(pilot_result())
    assert passed["status"] == "PASS"
    assert passed["metrics"]["bilateral_contact_max_last_50"] == 0.20
    assert passed["threshold_classification"] == "DEVELOPMENT_ONLY_DECISION"

    contact_edge = campaign.evaluate_pilot_gate(pilot_result(bilateral_contact_fraction=0.15))
    assert contact_edge["status"] == "FAIL"
    assert "PILOT_BILATERAL_CONTACT_GATE_FAILED" in contact_edge["reasons"]

    pushing = campaign.evaluate_pilot_gate(pilot_result(pushing_fraction=0.0011))
    assert pushing["status"] == "FAIL"
    assert "PILOT_PUSHING_GATE_FAILED" in pushing["reasons"]

    incomplete = pilot_result()
    incomplete["curve"] = incomplete["curve"][:40]
    assert campaign.evaluate_pilot_gate(incomplete)["status"] == "INVALID"

    missing_metric = pilot_result()
    for record in missing_metric["curve"]:
        record.pop("verify_fraction")
    assert campaign.evaluate_pilot_gate(missing_metric)["status"] == "INVALID"

    nonfinite_runtime = campaign.evaluate_pilot_gate(pilot_result(nonfinite_fraction=1.0e-4))
    assert nonfinite_runtime["status"] == "INVALID"
    assert "PILOT_NONFINITE_RUNTIME_METRIC" in nonfinite_runtime["reasons"]


def test_bootstrap_bool_audit_is_derived_from_baseline_and_current_source(
    tmp_path: Path,
) -> None:
    instance = campaign.Campaign(SimpleNamespace(run_root=tmp_path, repository_root=ROOT))
    audit = instance._bootstrap_bool_audit()
    assert audit["status"] == "PASS"
    assert audit["bug_present_in_pre_redesign_commit"] is True
    assert audit["bug_fixed_in_implementation"] is True
    assert audit["forbidden_bug_expression_absent"] is True


def test_box_pushed_is_derived_from_observed_violation_evidence(tmp_path: Path) -> None:
    reach_root = tmp_path / "reachability_smoke"
    campaign.atomic_write_json(
        reach_root / "reachability_result.json",
        {"cases": [{"pushing_reasons": []}, {"pushing_reasons": ["box_translation"]}]},
    )
    evidence = campaign.box_motion_violation_evidence(tmp_path)
    assert evidence["observed"] is True
    assert evidence["artifacts"][0]["pushing_case_count"] == 1

    campaign.atomic_write_json(
        reach_root / "reachability_result.json",
        {"cases": [{"pushing_reasons": []}]},
    )
    assert campaign.box_motion_violation_evidence(tmp_path)["observed"] is False


def test_reachability_failure_requires_complete_sha_verified_diagnostic_package(
    tmp_path: Path,
) -> None:
    cases = [{"initial_gap_m": value} for value in campaign.REACHABILITY_GAPS_M]
    payload = {
        "status": "FAIL",
        "diagnostic_valid": True,
        "cases": cases,
        "diagnostic_replay": {"status": "INVALID"},
    }
    result = campaign.StageResult(
        "REACHABILITY_SMOKE", 0, tmp_path / "result.json", payload, True, "FAILED"
    )
    assert campaign.Campaign._reachability_valid(result) is False

    diagnostic = {"status": "PASS"}
    for key in ("front_video", "side_video", "terminal_front_image", "terminal_side_image"):
        path = tmp_path / key
        path.write_bytes(key.encode())
        diagnostic[key] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": campaign.sha256_file(path),
        }
    payload["diagnostic_replay"] = diagnostic
    assert campaign.Campaign._reachability_valid(result) is True


def test_training_valid_requires_zero_runner_rc() -> None:
    payload = pilot_result()
    failed_rc = campaign.StageResult(
        "PILOT_TRAINING", 2, Path("training_result.json"), payload, True, "RC_2"
    )
    zero_rc = campaign.StageResult(
        "PILOT_TRAINING", 0, Path("training_result.json"), payload, True, "PASS"
    )
    assert campaign.Campaign._training_valid(failed_rc, 200) is False
    assert campaign.Campaign._training_valid(zero_rc, 200) is True


def test_stage_order_and_runner_interfaces_are_frozen() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    ordered_tokens = (
        "self.reachability()",
        'self.train("pilot")',
        "evaluate_pilot_gate(",
        'self.train("formal")',
        "self.screening(",
        "self.video(",
        "self.finalize()",
    )
    execute_start = source.index("    def execute(self) -> int:")
    positions = [source.index(token, execute_start) for token in ordered_tokens]
    assert positions == sorted(positions)
    for runner in (
        "run_s2_03t_reachability_smoke.py",
        "train_s2_03t.py",
        "screen_s2_03t_redesign_checkpoints.py",
        "record_s2_03t_redesign_video.py",
    ):
        assert runner in source
    assert campaign.REACHABILITY_GAPS_M == (0.0, 0.005, 0.010, 0.030, 0.060)
    assert campaign.PILOT_ITERATIONS == 200
    assert campaign.FORMAL_ITERATIONS == 1200
    assert campaign.SCREENING_SEEDS == (42, 43, 44)
    assert campaign.CHECKPOINT_LIMIT == 6
    assert "import isaaclab" not in source.lower()
    assert "subprocess.Popen" in source
    assert "EXECUTION_SOURCE_WORKTREE_DIRTY" in source
    assert "EXECUTION_SOURCE_SHA_MISMATCH" in source
    assert "STAGED_CHANGES_PRESENT_AT_CAMPAIGN_START" in source
    assert '"training_seed": 42' in source
    assert '"evaluate_s2_03t_checkpoint.py"' in source
    assert '"--query-compute-apps=pid"' in source
    assert "fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)" in source
    assert "result.raw_rc == 0" in source


def test_small_report_builder_never_embeds_video_or_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "video_delivery").mkdir(parents=True)
    campaign.atomic_write_json(
        run / "reachability_smoke/reachability_result.json",
        {"cases": [{"pushing_reasons": []}]},
    )
    fake_repository = tmp_path / "repo"
    execution_sources = {}

    def add_json_artifact(relative: str, payload: dict) -> Path:
        path = fake_repository / relative
        campaign.atomic_write_json(path, payload)
        execution_sources[relative] = {
            "path": str(path),
            "sha256": campaign.sha256_file(path),
        }
        return path

    preserved_evidence = {}
    for relative, expected in campaign.PRESERVED_EVIDENCE_EXPECTATIONS.items():
        path = add_json_artifact(relative, expected)
        preserved_evidence[relative] = {
            "path": str(path),
            "sha256": campaign.sha256_file(path),
            "verified_expected_fields": expected,
        }
    bootstrap_path = fake_repository / campaign.BOOTSTRAP_SOURCE_RELATIVE
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_path.write_text(
        'active = not bool(getattr(self.env, "_s2_03t_bootstrap_mode", False))\n',
        encoding="utf-8",
    )
    execution_sources[campaign.BOOTSTRAP_SOURCE_RELATIVE] = {
        "path": str(bootstrap_path),
        "sha256": campaign.sha256_file(bootstrap_path),
    }
    add_json_artifact(
        "reports/stage2/s2_03t_reward_landscape_audit.json",
        {"status": "PASS", "current_reward_has_hover_local_optimum": "YES"},
    )
    add_json_artifact(
        "reports/stage2/s2_03t_redesign_resolved_config.json",
        {
            "status": "READY_FOR_RUNTIME_INTEGRATION",
            "formal_s2_03_gates_unchanged": True,
            "reward_landscape_v2": {"required_relations": {"safe_hold_dominates": True}},
        },
    )
    campaign.atomic_write_json(
        run / "manifest.json",
        {
            "schema_version": 1,
            "tmux_session": "test",
            "stage_records": [],
            "execution_sources": execution_sources,
            "preserved_evidence_verified": True,
            "preserved_evidence": preserved_evidence,
            "bootstrap_bool_audit": {
                "status": "PASS",
                "bug_present_in_pre_redesign_commit": True,
                "bug_fixed_in_implementation": True,
                "pre_redesign_commit": campaign.PRE_REDESIGN_EVIDENCE_COMMIT,
                "source_relative": campaign.BOOTSTRAP_SOURCE_RELATIVE,
                "source_path": str(bootstrap_path),
                "source_sha256": campaign.sha256_file(bootstrap_path),
                "forbidden_bug_expression_absent": True,
            },
        },
    )
    campaign.atomic_write_json(
        run / "campaign_outcome.json",
        {
            "schema_version": 1,
            "status": "FAIL",
            "primary_reason": "NO_QUALIFIED_CONTACT_POLICY",
            "new_policy_status": "FAIL",
            "implementation_branch": campaign.EXPECTED_BRANCH,
            "tmux_session": "test",
            "safety_boundaries": {
                "box_push_commanded": False,
                "box_motion_violation_observed": False,
                "box_pushed": False,
            },
        },
    )
    (run / "video_delivery/primary_front.mp4").write_bytes(b"not-real-video")
    (run / "video_delivery/primary_side.mp4").write_bytes(b"not-real-video")
    campaign.atomic_write_json(
        run / "video_delivery/video_result.json",
        {"status": "PASS", "visualization_complete": True},
    )
    reports = finalizer.build_reports(run, "b" * 40)
    finalizer.validate_report_payloads(reports)
    assert tuple(reports) == finalizer.REPORT_RELATIVE_PATHS
    summary = reports[Path("reports/stage2/s2_03t_redesign_campaign_summary.json")]
    assert summary["training_failure_preserved"] is True
    assert summary["old_no_qualified_contact_policy_preserved"] is True
    video = reports[Path("reports/stage2/s2_03t_redesign_video_manifest.json")]
    assert video["files"]["primary_front.mp4"]["size_bytes"] == 14
    assert video["files"]["primary_front.mp4"]["path"].endswith("primary_front.mp4")
    encoded = json.dumps({str(key): value for key, value in reports.items()})
    assert "not-real-video" not in encoded

    reward_relative = "reports/stage2/s2_03t_reward_landscape_audit.json"
    reward_path = fake_repository / reward_relative
    campaign.atomic_write_json(
        reward_path,
        {"status": "FAIL", "current_reward_has_hover_local_optimum": "NO"},
    )
    manifest_payload = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload["execution_sources"][reward_relative]["sha256"] = campaign.sha256_file(
        reward_path
    )
    campaign.atomic_write_json(run / "manifest.json", manifest_payload)
    try:
        finalizer.build_reports(run, "b" * 40)
    except finalizer.FinalizerError as exc:
        assert "REWARD_LANDSCAPE_AUDIT_INVALID" in str(exc)
    else:
        raise AssertionError("contradictory reward audit must invalidate finalization")


def test_finalizer_uses_exact_add_and_ordinary_push_only() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    assert 'run_git(repository, "add", "--", *(str(path) for path in report_paths))' in source
    assert '"push",\n        args.remote,' in source
    for forbidden in (
        '"add", "."',
        '"add", "-A"',
        '"commit", "-a"',
        '"push", "--force"',
        '"push", "--force-with-lease"',
        '"reset"',
        '"clean"',
        '"restore"',
        '"stash"',
        '"rebase"',
    ):
        assert forbidden not in source
    assert "PUSH_FAILED_REMOTE_MOVED" in source
    assert "PREEXISTING_54_SHA_PRESERVED" in source
    assert "ACTIVE_MANIFEST" in source and "third_party" in source
