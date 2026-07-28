#!/usr/bin/env python3
"""Pure-Python report, safety, commit, and ordinary-push finalizer for S2-03T."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from s2_03t_redesign_campaign import (
    BOOTSTRAP_SOURCE_RELATIVE,
    PRE_REDESIGN_EVIDENCE_COMMIT,
    PRESERVED_EVIDENCE_EXPECTATIONS,
    atomic_write_json,
    box_motion_violation_evidence,
    finite_tree,
    read_json,
    sha256_file,
    utc_timestamp,
)

SCHEMA_VERSION = 1
RESULT_COMMIT_MESSAGE = "stage2: record redesigned S2-03T contact training"
REPORT_RELATIVE_PATHS = (
    Path("reports/stage2/s2_03t_redesign_campaign_summary.json"),
    Path("reports/stage2/s2_03t_redesign_screening_summary.json"),
    Path("reports/stage2/s2_03t_redesign_video_manifest.json"),
    Path("reports/stage2/s2_03t_redesign_disposition.json"),
)
ACTIVE_MANIFEST = Path("configs/stage1/active_controller_manifest.json")


class FinalizerError(RuntimeError):
    pass


def run_git(
    repository: Path,
    *arguments: str,
    check: bool = True,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=check,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def git_text(repository: Path, *arguments: str, timeout: float = 120.0) -> str:
    return run_git(repository, *arguments, timeout=timeout).stdout.strip()


def remote_branch_sha(repository: Path, remote: str, branch: str) -> str | None:
    output = git_text(repository, "ls-remote", "--heads", remote, f"refs/heads/{branch}")
    rows = [line.split() for line in output.splitlines() if line.strip()]
    return rows[0][0] if rows else None


def path_kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_file():
        return "regular"
    if path.is_dir():
        return "directory"
    return "missing"


def preservation_audit(repository: Path, baseline_path: Path) -> dict[str, Any]:
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return {
            "status": "INVALID",
            "primary_reason": "PRESERVATION_BASELINE_UNREADABLE",
            "error": repr(exc),
        }
    if not isinstance(baseline, list) or len(baseline) != 54:
        return {
            "status": "INVALID",
            "primary_reason": "PRESERVATION_BASELINE_NOT_HISTORICAL_54",
            "entry_count": len(baseline) if isinstance(baseline, list) else None,
        }
    mismatches: list[dict[str, Any]] = []
    for expected in baseline:
        relative = Path(str(expected["path"]))
        path = repository / relative
        kind = path_kind(path)
        size = path.stat().st_size if kind in {"regular", "symlink"} else None
        digest = sha256_file(path) if kind == "regular" else None
        status_process = run_git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            str(relative),
        )
        status_lines = status_process.stdout.splitlines()
        git_status = status_lines[0][:2] if status_lines else ""
        actual = {"file_type": kind, "size_bytes": size, "sha256": digest, "git_status": git_status}
        expected_subset = {
            "file_type": expected.get("file_type"),
            "size_bytes": expected.get("size_bytes"),
            "sha256": expected.get("sha256"),
            "git_status": expected.get("git_status"),
        }
        if actual != expected_subset:
            mismatches.append(
                {"path": str(relative), "expected": expected_subset, "actual": actual}
            )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "primary_reason": "PREEXISTING_54_SHA_PRESERVED"
        if not mismatches
        else "PREEXISTING_54_SHA_MISMATCH",
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": sha256_file(baseline_path),
        "entry_count": len(baseline),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def maximum(curve: object, key: str) -> float | None:
    if not isinstance(curve, list):
        return None
    values: list[float] = []
    for item in curve:
        if not isinstance(item, Mapping) or item.get(key) is None:
            continue
        try:
            value = float(item[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return max(values, default=None)


def latest_value(curve: object, key: str) -> object:
    if not isinstance(curve, list):
        return None
    for item in reversed(curve):
        if isinstance(item, Mapping) and key in item:
            return item[key]
    return None


def sanitize_checkpoint(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    return {
        key: entry.get(key)
        for key in (
            "path",
            "sha256",
            "size_bytes",
            "iteration",
            "iteration_field",
            "selection_reason",
            "qualified_seed_count",
            "qualified",
        )
        if key in entry
    }


def file_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def manifest_source_path(manifest: Mapping[str, Any], relative: str) -> Path:
    sources = manifest.get("execution_sources")
    record = sources.get(relative) if isinstance(sources, Mapping) else None
    if not isinstance(record, Mapping):
        raise FinalizerError(f"MANIFEST_SOURCE_RECORD_MISSING:{relative}")
    path = Path(str(record.get("path", ""))).resolve()
    relative_parts = Path(relative).parts
    if (
        len(path.parts) < len(relative_parts)
        or path.parts[-len(relative_parts) :] != relative_parts
        or not path.is_file()
        or sha256_file(path) != record.get("sha256")
    ):
        raise FinalizerError(f"MANIFEST_SOURCE_SHA_MISMATCH:{relative}")
    return path


def manifest_json_artifact(manifest: Mapping[str, Any], relative: str) -> dict[str, Any]:
    payload = read_json(manifest_source_path(manifest, relative))
    if payload is None:
        raise FinalizerError(f"MANIFEST_JSON_ARTIFACT_INVALID:{relative}")
    return payload


def validate_preserved_evidence(manifest: Mapping[str, Any]) -> bool:
    records = manifest.get("preserved_evidence")
    if manifest.get("preserved_evidence_verified") is not True or not isinstance(records, Mapping):
        raise FinalizerError("PRESERVED_EVIDENCE_MANIFEST_INVALID")
    for relative, expected in PRESERVED_EVIDENCE_EXPECTATIONS.items():
        record = records.get(relative)
        source_path = manifest_source_path(manifest, relative)
        source_record = manifest["execution_sources"][relative]
        payload = read_json(source_path)
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != source_record.get("sha256")
            or Path(str(record.get("path", ""))).resolve() != source_path
            or record.get("verified_expected_fields") != expected
            or payload is None
            or any(payload.get(key) != value for key, value in expected.items())
        ):
            raise FinalizerError(f"PRESERVED_EVIDENCE_MISMATCH:{relative}")
    return True


def validated_static_audits(manifest: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = manifest.get("bootstrap_bool_audit")
    bootstrap_source = manifest_source_path(manifest, BOOTSTRAP_SOURCE_RELATIVE)
    if (
        not isinstance(bootstrap, Mapping)
        or bootstrap.get("status") != "PASS"
        or bootstrap.get("pre_redesign_commit") != PRE_REDESIGN_EVIDENCE_COMMIT
        or bootstrap.get("source_relative") != BOOTSTRAP_SOURCE_RELATIVE
        or Path(str(bootstrap.get("source_path", ""))).resolve() != bootstrap_source
        or bootstrap.get("source_sha256") != sha256_file(bootstrap_source)
        or bootstrap.get("bug_present_in_pre_redesign_commit") is not True
        or bootstrap.get("bug_fixed_in_implementation") is not True
        or bootstrap.get("forbidden_bug_expression_absent") is not True
    ):
        raise FinalizerError("BOOTSTRAP_BOOL_AUDIT_INVALID")
    reward = manifest_json_artifact(manifest, "reports/stage2/s2_03t_reward_landscape_audit.json")
    hover = reward.get("current_reward_has_hover_local_optimum")
    if reward.get("status") != "PASS" or hover not in {"YES", "NO", "UNKNOWN"}:
        raise FinalizerError("REWARD_LANDSCAPE_AUDIT_INVALID")
    resolved = manifest_json_artifact(
        manifest, "reports/stage2/s2_03t_redesign_resolved_config.json"
    )
    relations = resolved.get("reward_landscape_v2", {}).get("required_relations")
    if (
        resolved.get("status") != "READY_FOR_RUNTIME_INTEGRATION"
        or resolved.get("formal_s2_03_gates_unchanged") is not True
        or not isinstance(relations, Mapping)
        or not relations
        or not all(value is True for value in relations.values())
    ):
        raise FinalizerError("REDESIGN_RESOLVED_AUDIT_INVALID")
    return {
        "bootstrap_bool_bug_present": bootstrap["bug_present_in_pre_redesign_commit"],
        "bootstrap_bool_bug_fixed": bootstrap["bug_fixed_in_implementation"],
        "reward_landscape_audit": reward["status"],
        "old_reward_has_hover_local_optimum": hover,
    }


def build_video_manifest(run_root: Path, video_result: Mapping[str, Any] | None) -> dict[str, Any]:
    delivery = run_root / "video_delivery"
    expected_names = (
        "primary_front.mp4",
        "primary_side.mp4",
        "primary_visual_result.json",
        "supplemental_front.mp4",
        "supplemental_side.mp4",
        "supplemental_visual_result.json",
        "video_manifest.json",
        "README.txt",
    )
    files = {
        name: record
        for name in expected_names
        if (record := file_record(delivery / name)) is not None
    }
    archive = file_record(run_root / "S2_03T_TRAINED_POLICY_VIDEO_DELIVERY.tar.gz")
    return {
        "schema_version": SCHEMA_VERSION,
        "package": "S2_03T_TRAINED_POLICY_VIDEO_DELIVERY",
        "generated_at_utc": utc_timestamp(),
        "run_root": str(run_root),
        "status": None if video_result is None else video_result.get("status"),
        "primary_reason": None if video_result is None else video_result.get("primary_reason"),
        "primary_seed": 42,
        "primary_video_recorded": bool(
            (delivery / "primary_front.mp4").is_file() and (delivery / "primary_side.mp4").is_file()
        ),
        "supplemental_video_recorded": bool(
            (delivery / "supplemental_front.mp4").is_file()
            and (delivery / "supplemental_side.mp4").is_file()
        ),
        "files": files,
        "archive": archive,
        "large_artifacts_committed_to_git": False,
    }


def build_reports(run_root: Path, implementation_commit: str) -> dict[Path, dict[str, Any]]:
    manifest = read_json(run_root / "manifest.json") or {}
    outcome = read_json(run_root / "campaign_outcome.json") or {}
    reachability = read_json(run_root / "reachability_smoke/reachability_result.json")
    pilot = read_json(run_root / "pilot_training/training_result.json")
    pilot_gate = read_json(run_root / "pilot_gate.json")
    formal = read_json(run_root / "formal_training/training_result.json")
    screening = read_json(run_root / "checkpoint_screening/screening_result.json")
    video_result = read_json(run_root / "video_delivery/video_result.json")
    video_manifest = build_video_manifest(run_root, video_result)

    best_checkpoint = None if screening is None else screening.get("best_checkpoint")
    if isinstance(best_checkpoint, str):
        best_checkpoint = {
            "path": best_checkpoint,
            "sha256": screening.get("best_checkpoint_sha256"),
        }
    best_checkpoint = sanitize_checkpoint(best_checkpoint)
    screening_candidates = []
    if screening is not None:
        raw_candidates = screening.get("checkpoint_results", screening.get("candidates", []))
        if isinstance(raw_candidates, list):
            screening_candidates = [
                value for item in raw_candidates if (value := sanitize_checkpoint(item)) is not None
            ]

    preserved_evidence_verified = validate_preserved_evidence(manifest)
    static_audits = validated_static_audits(manifest)
    box_motion = box_motion_violation_evidence(run_root)
    safety_boundaries = outcome.get("safety_boundaries")
    if not isinstance(safety_boundaries, Mapping):
        raise FinalizerError("OUTCOME_SAFETY_BOUNDARIES_MISSING")
    if (
        safety_boundaries.get("box_push_commanded") is not False
        or safety_boundaries.get("box_motion_violation_observed") != box_motion["observed"]
        or safety_boundaries.get("box_pushed") != box_motion["observed"]
    ):
        raise FinalizerError("BOX_MOTION_OUTCOME_EVIDENCE_MISMATCH")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign": "S2_03T_REDESIGN_TRAIN_SCREEN_AND_VIDEO_CAMPAIGN",
        "classification": "PROJECT_EVIDENCE",
        "generated_at_utc": utc_timestamp(),
        "status": outcome.get("status", "INVALID"),
        "primary_reason": outcome.get("primary_reason", "CAMPAIGN_OUTCOME_MISSING"),
        "implementation_branch": outcome.get("implementation_branch"),
        "implementation_commit": implementation_commit,
        "result_commit": "RECORDED_POST_COMMIT_IN_RUN_ROOT_FINALIZER_RESULT",
        "run_root": str(run_root),
        "tmux_session": outcome.get("tmux_session", manifest.get("tmux_session")),
        "training_failure_preserved": preserved_evidence_verified,
        "old_no_qualified_contact_policy_preserved": preserved_evidence_verified,
        "bootstrap_bool_bug_present": static_audits["bootstrap_bool_bug_present"],
        "bootstrap_bool_bug_fixed": static_audits["bootstrap_bool_bug_fixed"],
        "reward_landscape_audit": static_audits["reward_landscape_audit"],
        "old_reward_has_hover_local_optimum": static_audits["old_reward_has_hover_local_optimum"],
        "new_action_name": "HYBRID_NOMINAL_NORMAL_APPROACH_PLUS_LEARNED_BILATERAL_NORMAL_CORRECTION",
        "new_action_dim": 2,
        "new_action_reachability_smoke": None
        if reachability is None
        else reachability.get("status"),
        "pilot": {
            "status": None if pilot is None else pilot.get("status"),
            "iterations_completed": None if pilot is None else pilot.get("completed_iterations"),
            "bilateral_contact_max": None
            if pilot is None
            else maximum(pilot.get("curve"), "bilateral_contact_fraction"),
            "verify_max": None if pilot is None else maximum(pilot.get("curve"), "verify_fraction"),
            "gate": pilot_gate,
        },
        "formal": {
            "status": None if formal is None else formal.get("status"),
            "iterations_completed": None if formal is None else formal.get("completed_iterations"),
            "env_count": None if formal is None else formal.get("num_envs"),
            "bilateral_contact_max": None
            if formal is None
            else maximum(formal.get("curve"), "bilateral_contact_fraction"),
            "verify_max": None
            if formal is None
            else maximum(formal.get("curve"), "verify_fraction"),
            "hold_max": None
            if formal is None
            else maximum(formal.get("curve"), "hold_success_fraction"),
            "highest_curriculum_level": None
            if formal is None
            else maximum(formal.get("curve"), "curriculum_level"),
        },
        "screening": {
            "status": None if screening is None else screening.get("status"),
            "episode_count": None
            if screening is None
            else screening.get("screening_episode_count"),
            "qualified_checkpoint_count": None
            if screening is None
            else screening.get("qualified_checkpoint_count"),
            "new_policy_status": outcome.get("new_policy_status"),
            "best_checkpoint": best_checkpoint,
        },
        "video": {
            "status": video_manifest["status"],
            "primary_recorded": video_manifest["primary_video_recorded"],
            "supplemental_recorded": video_manifest["supplemental_video_recorded"],
            "archive": video_manifest["archive"],
        },
        "safety_boundaries": safety_boundaries,
        "source_manifest_sha256": sha256_file(run_root / "manifest.json")
        if (run_root / "manifest.json").is_file()
        else None,
    }
    screening_summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "S2-03T_REDESIGN_CHECKPOINT_SCREENING",
        "classification": "PROJECT_EVIDENCE",
        "generated_at_utc": utc_timestamp(),
        "status": None if screening is None else screening.get("status"),
        "primary_reason": None if screening is None else screening.get("primary_reason"),
        "diagnostic_valid": None if screening is None else screening.get("diagnostic_valid"),
        "screening_seeds": None if screening is None else screening.get("screening_seeds"),
        "candidate_count": None if screening is None else screening.get("candidate_count"),
        "screening_episode_count": None
        if screening is None
        else screening.get("screening_episode_count"),
        "qualified_checkpoint_count": None
        if screening is None
        else screening.get("qualified_checkpoint_count"),
        "best_checkpoint": best_checkpoint,
        "candidates": screening_candidates,
        "authoritative_result_path": str(run_root / "checkpoint_screening/screening_result.json"),
        "authoritative_result_sha256": (
            sha256_file(run_root / "checkpoint_screening/screening_result.json")
            if (run_root / "checkpoint_screening/screening_result.json").is_file()
            else None
        ),
    }
    disposition = {
        "schema_version": SCHEMA_VERSION,
        "stage": "S2-03T_REDESIGN_DISPOSITION",
        "generated_at_utc": utc_timestamp(),
        "status": outcome.get("status", "INVALID"),
        "primary_reason": outcome.get("primary_reason", "CAMPAIGN_OUTCOME_MISSING"),
        "source_fact": [
            "Persisted runner/evaluator JSON is authoritative over post-result Kit teardown return codes.",
            "The formal S2-03 no-pushing safety gates were not changed by this campaign.",
        ],
        "project_evidence": {
            "reachability_status": None if reachability is None else reachability.get("status"),
            "pilot_gate_status": None if pilot_gate is None else pilot_gate.get("status"),
            "formal_training_status": None if formal is None else formal.get("status"),
            "new_policy_status": outcome.get("new_policy_status", "INVALID"),
            "video_status": video_manifest["status"],
        },
        "project_inference": (
            "The redesigned policy is eligible for research-lead review before formal Attach qualification."
            if outcome.get("status") == "PASS"
            else "The redesign did not produce evidence sufficient for promotion."
        ),
        "development_only_decision": [
            "2D bilateral normal-correction authority and its limits",
            "nominal approach speed/acceleration/jerk and IK damping",
            "reward weights and three-mode switching",
            "curriculum promotion thresholds",
            "pilot fall-explosion and long-term-saturation thresholds",
        ],
        "promotion_to_formal_attach_qualification": "RESEARCH_LEAD_REVIEW_REQUIRED",
        "pushing_authorized": False,
        "s2_04_authorized": False,
        "falcon_migration_decision": "NO",
        "next_step": "等待研究负责人查看训练后视频和 screening 证据，再决定是否进入 S2-03 正式 Attach qualification；不得自动开始 pushing 或 S2-04。",
    }
    return {
        REPORT_RELATIVE_PATHS[0]: summary,
        REPORT_RELATIVE_PATHS[1]: screening_summary,
        REPORT_RELATIVE_PATHS[2]: video_manifest,
        REPORT_RELATIVE_PATHS[3]: disposition,
    }


def validate_report_payloads(reports: Mapping[Path, Mapping[str, Any]]) -> None:
    if tuple(reports) != REPORT_RELATIVE_PATHS:
        raise FinalizerError("REPORT_PATH_SET_MISMATCH")
    for relative, payload in reports.items():
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise FinalizerError(f"REPORT_SCHEMA_VERSION_INVALID:{relative}")
        if not finite_tree(payload):
            raise FinalizerError(f"REPORT_NONFINITE:{relative}")
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
        if len(encoded) > 2 * 1024 * 1024:
            raise FinalizerError(f"REPORT_TOO_LARGE:{relative}:{len(encoded)}")
        forbidden_suffixes = (".pt", ".mp4", ".png", ".tar.gz", ".jsonl", ".log")
        if any(str(relative).endswith(suffix) for suffix in forbidden_suffixes):
            raise FinalizerError(f"LARGE_ARTIFACT_REPORT_PATH:{relative}")


def verify_repository_boundaries(
    repository: Path,
    baseline: Path,
    implementation_commit: str,
    expected_branch: str,
) -> dict[str, Any]:
    branch = git_text(repository, "branch", "--show-current")
    head = git_text(repository, "rev-parse", "HEAD")
    staged = [
        line
        for line in git_text(repository, "diff", "--cached", "--name-only").splitlines()
        if line
    ]
    preservation = preservation_audit(repository, baseline)
    third_party_status = git_text(
        repository, "status", "--porcelain=v1", "--untracked-files=all", "--", "third_party"
    )
    third_party_diff = git_text(
        repository, "diff", "--name-only", implementation_commit, "--", "third_party"
    )
    active_staged = git_text(
        repository, "diff", "--cached", "--name-only", "--", str(ACTIVE_MANIFEST)
    )
    active_record = next(
        (
            mismatch
            for mismatch in preservation.get("mismatches", [])
            if mismatch.get("path") == str(ACTIVE_MANIFEST)
        ),
        None,
    )
    checks = {
        "branch_matches": branch == expected_branch,
        "head_matches_implementation": head == implementation_commit,
        "staged_files_empty_before_finalizer": not staged,
        "preexisting_54_sha_preserved": preservation.get("status") == "PASS",
        "third_party_unmodified": not third_party_status and not third_party_diff,
        "active_manifest_not_updated": active_record is None and not active_staged,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "primary_reason": "FINALIZER_REPOSITORY_BOUNDARIES_PASSED"
        if all(checks.values())
        else "FINALIZER_REPOSITORY_BOUNDARY_FAILED",
        "branch": branch,
        "head": head,
        "implementation_commit": implementation_commit,
        "staged_files": staged,
        "preservation": preservation,
        "third_party_status": third_party_status,
        "third_party_diff_from_implementation": third_party_diff,
        "active_manifest_staged": bool(active_staged),
        "checks": checks,
    }


def exact_stage_and_commit(
    repository: Path,
    report_paths: Sequence[Path],
) -> str:
    run_git(repository, "add", "--", *(str(path) for path in report_paths))
    staged = tuple(
        Path(line)
        for line in git_text(repository, "diff", "--cached", "--name-only").splitlines()
        if line
    )
    if staged != tuple(sorted(report_paths, key=str)):
        raise FinalizerError(
            "CACHED_PATH_SET_MISMATCH:"
            + json.dumps(
                {
                    "expected": [str(path) for path in report_paths],
                    "actual": [str(path) for path in staged],
                }
            )
        )
    check = run_git(repository, "diff", "--cached", "--check", check=False)
    if check.returncode != 0:
        raise FinalizerError(f"CACHED_DIFF_CHECK_FAILED:{check.stdout}:{check.stderr}")
    commit = run_git(repository, "commit", "-m", RESULT_COMMIT_MESSAGE, check=False, timeout=180.0)
    if commit.returncode != 0:
        raise FinalizerError(f"RESULT_COMMIT_FAILED:{commit.stdout}:{commit.stderr}")
    return git_text(repository, "rev-parse", "HEAD")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    repository = args.repository_root.resolve()
    if run_root == repository or repository in run_root.parents:
        raise FinalizerError("RUN_ROOT_MUST_BE_OUTSIDE_REPOSITORY")
    if not (run_root / "campaign_outcome.json").is_file():
        raise FinalizerError("CAMPAIGN_OUTCOME_MISSING")
    if list(run_root.rglob("model_1999.pt")):
        raise FinalizerError("MODEL_1999_PRESENT_IN_CAMPAIGN")

    pre = verify_repository_boundaries(
        repository,
        args.preservation_baseline.resolve(),
        args.implementation_commit,
        args.branch,
    )
    if pre["status"] != "PASS":
        raise FinalizerError(f"PRE_COMMIT_SAFETY_AUDIT_FAILED:{pre['primary_reason']}")

    reports = build_reports(run_root, args.implementation_commit)
    validate_report_payloads(reports)
    for relative, payload in reports.items():
        atomic_write_json(repository / relative, payload)
        # A parse round-trip is the final pure-static schema/JSON check.
        loaded = json.loads((repository / relative).read_text(encoding="utf-8"))
        if loaded != payload:
            raise FinalizerError(f"REPORT_ROUND_TRIP_MISMATCH:{relative}")

    result_commit = exact_stage_and_commit(repository, tuple(sorted(reports, key=str)))
    push = run_git(
        repository,
        "push",
        args.remote,
        f"HEAD:refs/heads/{args.branch}",
        check=False,
        timeout=300.0,
    )
    remote_after = remote_branch_sha(repository, args.remote, args.branch)
    upstream_after: str | None
    try:
        upstream_after = git_text(repository, "rev-parse", "@{upstream}")
    except subprocess.CalledProcessError:
        upstream_after = None
    local_after = git_text(repository, "rev-parse", "HEAD")
    if push.returncode != 0:
        reason = (
            "PUSH_FAILED_REMOTE_MOVED"
            if remote_after not in {None, args.implementation_commit, result_commit}
            else "PUSH_FAILED"
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "primary_reason": reason,
            "result_commit": result_commit,
            "push_rc": push.returncode,
            "push_stdout": push.stdout,
            "push_stderr": push.stderr,
            "local_sha": local_after,
            "upstream_sha": upstream_after,
            "remote_sha": remote_after,
            "all_code_commits_pushed": False,
            "all_remote_sha_verified": False,
            "force_push_used": False,
            "rebase_used": False,
        }
    verified = local_after == upstream_after == remote_after == result_commit
    post_preservation = preservation_audit(repository, args.preservation_baseline.resolve())
    post_staged = git_text(repository, "diff", "--cached", "--name-only")
    if post_preservation.get("status") != "PASS" or post_staged:
        verified = False
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if verified else "FAIL",
        "primary_reason": "RESULT_COMMIT_PUSHED_AND_VERIFIED"
        if verified
        else "POST_PUSH_VERIFICATION_FAILED",
        "result_commit": result_commit,
        "push_rc": push.returncode,
        "local_sha": local_after,
        "upstream_sha": upstream_after,
        "remote_sha": remote_after,
        "all_code_commits_pushed": verified,
        "all_remote_sha_verified": verified,
        "preexisting_54_sha_preserved": post_preservation.get("status") == "PASS",
        "preexisting_dirty_files_staged": False,
        "active_manifest_not_updated": True,
        "third_party_unmodified": True,
        "force_push_used": False,
        "force_with_lease_used": False,
        "rebase_used": False,
        "reports_committed": [str(path) for path in REPORT_RELATIVE_PATHS],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--preservation-baseline", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--remote", default="origin")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result_path = args.run_root.resolve() / "finalizer_result.json"
    try:
        result = finalize(args)
    except BaseException as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "INVALID",
            "primary_reason": "FINALIZER_EXCEPTION",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "result_commit": None,
            "all_code_commits_pushed": False,
            "all_remote_sha_verified": False,
            "force_push_used": False,
            "rebase_used": False,
        }
    atomic_write_json(result_path, result)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
