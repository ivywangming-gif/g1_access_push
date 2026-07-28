#!/usr/bin/env python3
"""Crash-resilient, single-process orchestrator for the S2-03T redesign campaign.

This module never imports Isaac Lab.  It launches exactly one stage process at a
time, treats complete JSON artifacts as authoritative, and keeps enough atomic
state on disk for an interrupted SSH/Codex session to be audited later.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
HEARTBEAT_INTERVAL_SECONDS = 30
PILOT_ITERATIONS = 200
FORMAL_ITERATIONS = 1200
PILOT_ENV_COUNT = 64
FORMAL_ENV_COUNT = 256
CHECKPOINT_LIMIT = 6
SCREENING_SEEDS = (42, 43, 44)
REACHABILITY_GAPS_M = (0.0, 0.005, 0.010, 0.030, 0.060)
EXPECTED_BRANCH = "stage2/s2-03t-reward-action-redesign"
EXPECTED_REMOTE_URL = "ssh://git@ssh.github.com:443/ivywangming-gif/g1_access_push.git"
LOCK_PATH = Path("/tmp/g1_access_push_s2_03t_redesign_campaign.global.flock")
PRE_REDESIGN_EVIDENCE_COMMIT = "8bef297b38051bb9b550e6e97bc7f3ee6370a635"
BOOTSTRAP_SOURCE_RELATIVE = "src/g1_access_push/sim/stage2/s2_03t_mdp.py"
EXECUTION_SOURCE_RELATIVES = (
    "configs/stage2/s2_03t_contact_action_contract.yaml",
    "configs/stage2/s2_03t_contact_curriculum_reward_contract.yaml",
    "configs/stage2/s2_03t_contact_curriculum_contract.yaml",
    "reports/stage2/s2_03t_redesign_resolved_config.json",
    "reports/stage2/s2_03t_reward_landscape_audit.json",
    "reports/stage2/s2_03t_reward_landscape_audit.md",
    "reports/stage2/s2_03t_action_authority_summary.json",
    "reports/stage2/s2_03t_screening_summary.json",
    "reports/stage2/s2_03t_visual_evidence_summary.json",
    "configs/stage2/s2_03_attach_only.yaml",
    "src/g1_access_push/sim/stage1/no_box_env_cfg.py",
    "src/g1_access_push/sim/stage2/s2_01_box_env.py",
    "src/g1_access_push/sim/stage2/s2_03_attach_env.py",
    "src/g1_access_push/sim/stage2/s2_03t_eval_cfg.py",
    "src/g1_access_push/stage2/s2_02_contract.py",
    "src/g1_access_push/stage2/s2_03_contract.py",
    "src/g1_access_push/stage2/s2_03t_contact_filter_contract.py",
    "src/g1_access_push/stage2/s2_03t_tensor_contract.py",
    "scripts/stage2/audit_s2_03t_reward_landscape.py",
    "scripts/stage2/finalize_s2_03t_redesign_campaign.py",
    "scripts/stage2/generate_s2_03t_redesign_resolved_config.py",
    "scripts/stage2/run_s2_03t_redesign_campaign.sh",
    "scripts/stage2/s2_03t_redesign_campaign.py",
    "scripts/stage2_isaac/evaluate_s2_03t_checkpoint.py",
    "scripts/stage2_isaac/record_s2_03t_redesign_video.py",
    "scripts/stage2_isaac/run_s2_03t_reachability_smoke.py",
    "scripts/stage2_isaac/s2_03t_visual_recorder.py",
    "scripts/stage2_isaac/screen_s2_03t_redesign_checkpoints.py",
    "scripts/stage2_isaac/train_s2_03t.py",
    "src/g1_access_push/sim/stage2/s2_03t_actions.py",
    "src/g1_access_push/sim/stage2/s2_03t_agent_cfg.py",
    "src/g1_access_push/sim/stage2/s2_03t_bootstrap.py",
    "src/g1_access_push/sim/stage2/s2_03t_env.py",
    "src/g1_access_push/sim/stage2/s2_03t_env_cfg.py",
    "src/g1_access_push/sim/stage2/s2_03t_mdp.py",
    "src/g1_access_push/sim/stage2/s2_03t_visual_cfg.py",
    "src/g1_access_push/stage2/s2_03t_redesign_contract.py",
)
FROZEN_CONTRACT_RELATIVES = EXECUTION_SOURCE_RELATIVES[:5]
PRESERVED_EVIDENCE_EXPECTATIONS = {
    "reports/stage2/s2_03t_screening_summary.json": {
        "status": "FAIL",
        "primary_reason": "NO_QUALIFIED_CONTACT_POLICY",
        "observed_episode_count": 18,
        "valid_episode_count": 18,
        "qualified_screening_checkpoints": 0,
    },
    "reports/stage2/s2_03t_action_authority_summary.json": {
        "audit_status": "PASS",
        "authority_status": "INSUFFICIENT_FOR_CONTACT",
        "primary_diagnosis": "ARM_RESIDUAL_AUTHORITY_INSUFFICIENT",
    },
    "reports/stage2/s2_03t_visual_evidence_summary.json": {
        "visualization_package_status": "PASS",
        "frozen_scientific_status": "FAIL",
        "frozen_scientific_primary_reason": "NO_QUALIFIED_CONTACT_POLICY",
    },
}


class CampaignError(RuntimeError):
    """Failure that should terminate the automatic campaign safely."""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, value: str) -> None:
    """Write and rename with both file and parent-directory durability."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def append_jsonl(path: Path, payload: object) -> None:
    """Append one complete JSON object with one O_APPEND write and fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short append to {path}: {written}/{len(encoded)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def command_output(command: Sequence[str], cwd: Path, timeout: float = 30.0) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_checkpoint(run_root: Path) -> str | None:
    candidates = [path for path in run_root.rglob("model_*.pt") if path.name != "model_1999.pt"]
    if not candidates:
        return None
    selected = max(
        candidates, key=lambda item: (checkpoint_iteration(item), item.stat().st_mtime_ns)
    )
    return str(selected.resolve())


def box_motion_violation_evidence(run_root: Path) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    observed = False

    reachability = read_json(run_root / "reachability_smoke/reachability_result.json")
    if reachability is not None:
        cases = reachability.get("cases")
        pushing_case_count = (
            sum(bool(case.get("pushing_reasons")) for case in cases if isinstance(case, Mapping))
            if isinstance(cases, list)
            else 0
        )
        evidence.append({"artifact": "reachability", "pushing_case_count": pushing_case_count})
        observed = observed or pushing_case_count > 0

    for stage_name in ("pilot_training", "formal_training"):
        training = read_json(run_root / stage_name / "training_result.json")
        if training is None:
            continue
        curve = training.get("curve")
        fractions = []
        if isinstance(curve, list):
            for record in curve:
                if not isinstance(record, Mapping):
                    continue
                try:
                    value = float(record.get("pushing_fraction", 0.0))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    fractions.append(value)
        maximum_fraction = max(fractions, default=0.0)
        evidence.append({"artifact": stage_name, "maximum_pushing_fraction": maximum_fraction})
        observed = observed or maximum_fraction > 0.0

    screening = read_json(run_root / "checkpoint_screening/screening_result.json")
    if screening is not None:
        checkpoints = screening.get("checkpoint_results")
        pushing_episode_count = 0
        if isinstance(checkpoints, list):
            for checkpoint in checkpoints:
                episodes = checkpoint.get("episodes") if isinstance(checkpoint, Mapping) else None
                if isinstance(episodes, list):
                    pushing_episode_count += sum(
                        bool(episode.get("pushing_reasons"))
                        for episode in episodes
                        if isinstance(episode, Mapping)
                    )
        evidence.append(
            {"artifact": "checkpoint_screening", "pushing_episode_count": pushing_episode_count}
        )
        observed = observed or pushing_episode_count > 0

    return {
        "observed": observed if evidence else None,
        "evidence_artifact_count": len(evidence),
        "artifacts": evidence,
    }


def evaluate_pilot_gate(
    result: Mapping[str, Any], expected_iterations: int = PILOT_ITERATIONS
) -> dict[str, Any]:
    """Apply the one-shot frozen pilot gate to the final 50 iterations."""

    reasons: list[str] = []
    status = result.get("status")
    completed = int(result.get("completed_iterations", -1))
    curve = result.get("curve")
    checkpoints = result.get("checkpoints")
    if status != "PASS":
        reasons.append("PILOT_TRAINING_NOT_COMPLETE")
    if completed != expected_iterations:
        reasons.append("PILOT_ITERATION_COUNT_MISMATCH")
    if result.get("finite") is not True or not finite_tree(result):
        reasons.append("PILOT_NONFINITE")
    if not isinstance(checkpoints, list) or not checkpoints:
        reasons.append("PILOT_CHECKPOINT_MISSING")
    if not isinstance(curve, list) or len(curve) < 50:
        reasons.append("PILOT_LAST_50_METRICS_MISSING")
        last_fifty: list[Mapping[str, Any]] = []
    else:
        last_fifty = [item for item in curve[-50:] if isinstance(item, Mapping)]
        if len(last_fifty) != 50:
            reasons.append("PILOT_LAST_50_METRICS_MALFORMED")

    metric_names = (
        "bilateral_contact_fraction",
        "verify_fraction",
        "hard_safety_violation_fraction",
        "pushing_fraction",
        "fall_fraction",
        "nonfinite_fraction",
        "action_saturation_fraction",
    )
    values: dict[str, list[float]] = {}
    for name in metric_names:
        try:
            series = [float(item[name]) for item in last_fifty]
        except (KeyError, TypeError, ValueError):
            series = []
            if last_fifty:
                reasons.append(f"PILOT_METRIC_MISSING:{name}")
        if series and not all(math.isfinite(value) for value in series):
            reasons.append(f"PILOT_NONFINITE_METRIC:{name}")
        values[name] = series

    metrics: dict[str, float | None] = {
        "bilateral_contact_max_last_50": max(values["bilateral_contact_fraction"], default=None),
        "verify_max_last_50": max(values["verify_fraction"], default=None),
        "hard_safety_max_last_50": max(values["hard_safety_violation_fraction"], default=None),
        "pushing_max_last_50": max(values["pushing_fraction"], default=None),
        "fall_max_last_50": max(values["fall_fraction"], default=None),
        "nonfinite_max_last_50": max(values["nonfinite_fraction"], default=None),
        "action_saturation_mean_last_50": (
            sum(values["action_saturation_fraction"]) / len(values["action_saturation_fraction"])
            if values["action_saturation_fraction"]
            else None
        ),
    }
    if (
        metrics["bilateral_contact_max_last_50"] is not None
        and metrics["bilateral_contact_max_last_50"] <= 0.15
    ):
        reasons.append("PILOT_BILATERAL_CONTACT_GATE_FAILED")
    if metrics["verify_max_last_50"] is not None and metrics["verify_max_last_50"] <= 0.05:
        reasons.append("PILOT_VERIFY_GATE_FAILED")
    if (
        metrics["hard_safety_max_last_50"] is not None
        and metrics["hard_safety_max_last_50"] >= 0.02
    ):
        reasons.append("PILOT_HARD_SAFETY_GATE_FAILED")
    if metrics["pushing_max_last_50"] is not None and metrics["pushing_max_last_50"] > 0.001:
        reasons.append("PILOT_PUSHING_GATE_FAILED")
    # These two thresholds are explicit development-only choices for the qualitative
    # "no fall explosion" and "not long-term saturated" requirements.
    if metrics["fall_max_last_50"] is not None and metrics["fall_max_last_50"] > 0.10:
        reasons.append("PILOT_FALL_EXPLOSION")
    if metrics["nonfinite_max_last_50"] is not None and metrics["nonfinite_max_last_50"] != 0.0:
        reasons.append("PILOT_NONFINITE_RUNTIME_METRIC")
    if (
        metrics["action_saturation_mean_last_50"] is not None
        and metrics["action_saturation_mean_last_50"] >= 0.95
    ):
        reasons.append("PILOT_ACTION_LONG_TERM_SATURATION")

    evidence_complete = not any(
        "MISSING" in reason
        or "MALFORMED" in reason
        or "NONFINITE" in reason
        or "COUNT_MISMATCH" in reason
        for reason in reasons
    )
    passed = not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "PILOT_GATE",
        "status": "PASS" if passed else ("FAIL" if evidence_complete else "INVALID"),
        "primary_reason": "PILOT_AUTOMATIC_GATE_PASSED" if passed else reasons[0],
        "reasons": reasons,
        "expected_iterations": expected_iterations,
        "last_iteration_window": 50,
        "metrics": metrics,
        "thresholds": {
            "bilateral_contact_fraction_strictly_greater_than": 0.15,
            "verify_fraction_strictly_greater_than": 0.05,
            "hard_safety_violation_fraction_strictly_less_than": 0.02,
            "pushing_fraction_less_than_or_equal_to": 0.001,
            "fall_explosion_fraction_less_than_or_equal_to": 0.10,
            "nonfinite_fraction_equal_to": 0.0,
            "mean_action_saturation_fraction_strictly_less_than": 0.95,
        },
        "threshold_classification": "DEVELOPMENT_ONLY_DECISION",
        "evaluated_at_utc": utc_timestamp(),
    }


@dataclass(frozen=True)
class StageResult:
    stage: str
    raw_rc: int
    artifact: Path
    payload: dict[str, Any] | None
    authoritative: bool
    reason: str


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_root = args.run_root.resolve()
        self.repository = args.repository_root.resolve()
        self.status_path = self.run_root / "status.json"
        self.heartbeat_path = self.run_root / "heartbeat.json"
        self.history_path = self.run_root / "stage_history.jsonl"
        self.manifest_path = self.run_root / "manifest.json"
        self.outcome_path = self.run_root / "campaign_outcome.json"
        self.process_rc_root = self.run_root / "process_rc"
        self.stop_heartbeat = threading.Event()
        self.state_lock = threading.Lock()
        self.child: subprocess.Popen[bytes] | None = None
        self.stage = "STARTING"
        self.command = "INITIALIZE"
        self.failure_reason: str | None = None
        self.started_monotonic = time.monotonic()
        self.heartbeat_thread: threading.Thread | None = None
        self.interrupted_signal: int | None = None
        self.stage_records: list[dict[str, Any]] = []

    def _git(self, *arguments: str) -> str:
        return command_output(("git", *arguments), self.repository)

    def _verify_execution_sources(self) -> None:
        if self._git("rev-parse", "HEAD") != self.args.implementation_commit:
            raise CampaignError("IMPLEMENTATION_HEAD_CHANGED_DURING_CAMPAIGN")
        if self._git("diff", "--cached", "--name-only"):
            raise CampaignError("STAGED_CHANGES_PRESENT_DURING_CAMPAIGN")
        target_status = self._git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *EXECUTION_SOURCE_RELATIVES,
        )
        if target_status:
            raise CampaignError(f"EXECUTION_SOURCE_WORKTREE_DIRTY:{target_status}")
        manifest = read_json(self.manifest_path)
        records = None if manifest is None else manifest.get("execution_sources")
        if not isinstance(records, Mapping) or set(records) != set(EXECUTION_SOURCE_RELATIVES):
            raise CampaignError("EXECUTION_SOURCE_MANIFEST_INVALID")
        for relative in EXECUTION_SOURCE_RELATIVES:
            record = records.get(relative)
            path = self.repository / relative
            if (
                not isinstance(record, Mapping)
                or Path(str(record.get("path", ""))).resolve() != path.resolve()
                or not path.is_file()
                or sha256_file(path) != record.get("sha256")
            ):
                raise CampaignError(f"EXECUTION_SOURCE_SHA_MISMATCH:{relative}")

    def _bootstrap_bool_audit(self) -> dict[str, Any]:
        baseline_source = self._git(
            "show", f"{PRE_REDESIGN_EVIDENCE_COMMIT}:{BOOTSTRAP_SOURCE_RELATIVE}"
        )
        current_path = self.repository / BOOTSTRAP_SOURCE_RELATIVE
        current_source = current_path.read_text(encoding="utf-8")
        bug_present = "active = ~getattr(" in baseline_source
        fixed_expression = 'active = not bool(getattr(self.env, "_s2_03t_bootstrap_mode", False))'
        bug_fixed = (
            fixed_expression in current_source and "active = ~getattr(" not in current_source
        )
        status = "PASS" if bug_present and bug_fixed else "FAIL"
        return {
            "status": status,
            "bug_present_in_pre_redesign_commit": bug_present,
            "bug_fixed_in_implementation": bug_fixed,
            "pre_redesign_commit": PRE_REDESIGN_EVIDENCE_COMMIT,
            "source_relative": BOOTSTRAP_SOURCE_RELATIVE,
            "source_path": str(current_path),
            "source_sha256": sha256_file(current_path),
            "required_fixed_expression": fixed_expression,
            "forbidden_bug_expression_absent": "active = ~getattr(" not in current_source,
            "classification": "STATIC_SOURCE_AUDIT",
        }

    def _initial_process_audit(self) -> list[dict[str, Any]]:
        output = command_output(("ps", "-eo", "pid=,ppid=,etimes=,args="), self.repository)
        own = {os.getpid(), os.getppid()}
        tokens = (
            "isaac-sim",
            "kit/kit",
            "SimulationApp",
            "s2_03t_redesign_campaign.py",
            "evaluate_s2_03t_checkpoint.py",
            "run_s2_03t_reachability_smoke.py",
            "train_s2_03t.py",
            "screen_s2_03t_redesign_checkpoints.py",
            "record_s2_03t_redesign_video.py",
        )
        matches: list[dict[str, Any]] = []
        for line in output.splitlines():
            fields = line.strip().split(maxsplit=3)
            if len(fields) < 4:
                continue
            try:
                pid, ppid = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            command = fields[3]
            if pid in own or ppid in own:
                continue
            if any(token in command for token in tokens):
                matches.append({"pid": pid, "ppid": ppid, "command": command})
        try:
            gpu_output = command_output(
                (
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ),
                self.repository,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            gpu_output = ""
        known_pids = {int(item["pid"]) for item in matches}
        for row in gpu_output.splitlines():
            try:
                pid = int(row.strip())
            except ValueError:
                continue
            if pid in own or pid in known_pids:
                continue
            try:
                command = command_output(("ps", "-p", str(pid), "-o", "args="), self.repository)
            except subprocess.SubprocessError:
                continue
            if any(token in command for token in tokens) or str(self.repository) in command:
                matches.append(
                    {"pid": pid, "ppid": None, "command": command, "source": "nvidia-smi"}
                )
        return matches

    def _verify_launcher_lock(self) -> None:
        try:
            inherited_target = Path(os.readlink("/proc/self/fd/9")).resolve()
            fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (FileNotFoundError, OSError) as exc:
            raise CampaignError("CAMPAIGN_GLOBAL_FLOCK_NOT_HELD") from exc
        if inherited_target != LOCK_PATH.resolve():
            raise CampaignError(f"CAMPAIGN_GLOBAL_FLOCK_PATH_MISMATCH:{inherited_target}")

    def initialize(self) -> None:
        self._verify_launcher_lock()
        self.process_rc_root.mkdir(parents=True, exist_ok=True)
        branch = self._git("branch", "--show-current")
        local_head = self._git("rev-parse", "HEAD")
        remote_url = self._git("remote", "get-url", "origin")
        upstream_head = self._git("rev-parse", "@{upstream}")
        remote_rows = self._git(
            "ls-remote", "--heads", "origin", f"refs/heads/{EXPECTED_BRANCH}"
        ).splitlines()
        remote_head = remote_rows[0].split()[0] if remote_rows else None
        reference = self.args.reference.resolve()
        if branch != EXPECTED_BRANCH:
            raise CampaignError(f"BRANCH_MISMATCH:{branch}")
        if local_head != self.args.implementation_commit:
            raise CampaignError(f"IMPLEMENTATION_HEAD_MISMATCH:{local_head}")
        if remote_url != EXPECTED_REMOTE_URL:
            raise CampaignError(f"REMOTE_URL_MISMATCH:{remote_url}")
        if upstream_head != local_head or remote_head != local_head:
            raise CampaignError(
                f"IMPLEMENTATION_NOT_PUSHED_OR_REMOTE_MISMATCH:{local_head}:{upstream_head}:{remote_head}"
            )
        if not reference.is_file() or sha256_file(reference) != self.args.reference_sha256:
            raise CampaignError("PRECONTACT_REFERENCE_SHA_MISMATCH")
        if not self.args.preservation_baseline.is_file():
            raise CampaignError("HISTORICAL_54_PRESERVATION_BASELINE_MISSING")
        baseline = json.loads(self.args.preservation_baseline.read_text(encoding="utf-8"))
        if not isinstance(baseline, list) or len(baseline) != 54:
            raise CampaignError("HISTORICAL_54_PRESERVATION_BASELINE_INVALID")
        active_processes = self._initial_process_audit()
        if active_processes:
            atomic_write_json(
                self.run_root / "startup_process_conflict.json", {"processes": active_processes}
            )
            raise CampaignError("ACTIVE_ISAAC_OR_S2_03T_PROCESS_PRESENT")

        staged_paths = self._git("diff", "--cached", "--name-only")
        if staged_paths:
            raise CampaignError(f"STAGED_CHANGES_PRESENT_AT_CAMPAIGN_START:{staged_paths}")
        target_status = self._git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *EXECUTION_SOURCE_RELATIVES,
        )
        if target_status:
            raise CampaignError(f"EXECUTION_SOURCE_WORKTREE_DIRTY:{target_status}")
        tracked_sources = set(self._git("ls-files", "--", *EXECUTION_SOURCE_RELATIVES).splitlines())
        if tracked_sources != set(EXECUTION_SOURCE_RELATIVES):
            raise CampaignError("EXECUTION_SOURCE_NOT_TRACKED_BY_IMPLEMENTATION_COMMIT")
        execution_paths = [self.repository / relative for relative in EXECUTION_SOURCE_RELATIVES]
        if not all(path.is_file() for path in execution_paths):
            raise CampaignError("EXECUTION_SOURCE_MISSING")
        contract_paths = [self.repository / relative for relative in FROZEN_CONTRACT_RELATIVES]
        if not all(path.is_file() for path in contract_paths):
            raise CampaignError("FROZEN_REDESIGN_CONTRACT_MISSING")
        bootstrap_bool_audit = self._bootstrap_bool_audit()
        if bootstrap_bool_audit["status"] != "PASS":
            raise CampaignError("BOOTSTRAP_BOOL_STATIC_AUDIT_FAILED")
        preserved_evidence = {}
        for relative, expected in PRESERVED_EVIDENCE_EXPECTATIONS.items():
            path = self.repository / relative
            payload = read_json(path)
            if payload is None or any(payload.get(key) != value for key, value in expected.items()):
                raise CampaignError(f"PRESERVED_S2_03T_EVIDENCE_MISMATCH:{relative}")
            preserved_evidence[relative] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "verified_expected_fields": expected,
            }

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "campaign": "S2_03T_REDESIGN_TRAIN_SCREEN_AND_VIDEO_CAMPAIGN",
            "created_at_utc": utc_timestamp(),
            "run_root": str(self.run_root),
            "campaign_log": str(self.run_root / "campaign.console.log"),
            "tmux_session": self.args.tmux_session,
            "main_pid": os.getpid(),
            "branch": branch,
            "implementation_commit": local_head,
            "upstream_sha": upstream_head,
            "remote_sha": remote_head,
            "remote_url": remote_url,
            "reference": {"path": str(reference), "sha256": self.args.reference_sha256},
            "preservation_baseline": {
                "path": str(self.args.preservation_baseline.resolve()),
                "entry_count": 54,
                "sha256": sha256_file(self.args.preservation_baseline),
            },
            "preserved_evidence_verified": True,
            "bootstrap_bool_audit": bootstrap_bool_audit,
            "preserved_evidence": preserved_evidence,
            "execution_sources": {
                relative: {
                    "path": str(self.repository / relative),
                    "sha256": sha256_file(self.repository / relative),
                }
                for relative in EXECUTION_SOURCE_RELATIVES
            },
            "frozen_contracts": {
                str(path.relative_to(self.repository)): {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in contract_paths
            },
            "duplicate_count": 0,
            "stage_order": [
                "REACHABILITY_SMOKE",
                "PILOT_TRAINING",
                "PILOT_GATE",
                "FORMAL_TRAINING",
                "CHECKPOINT_SCREENING",
                "TRAINED_POLICY_VIDEO",
                "FINALIZER",
            ],
            "frozen_execution": {
                "action_name": (
                    "HYBRID_NOMINAL_NORMAL_APPROACH_PLUS_"
                    "LEARNED_BILATERAL_NORMAL_CORRECTION"
                ),
                "action_dim": 2,
                "reachability_gaps_m": list(REACHABILITY_GAPS_M),
                "pilot_env_count": PILOT_ENV_COUNT,
                "pilot_iterations": PILOT_ITERATIONS,
                "pilot_curriculum_max_level": 1,
                "formal_env_count": FORMAL_ENV_COUNT,
                "formal_env_count_reason": "PRIOR_256_ENV_S2_03T_RUN_COMPLETED_ON_THIS_MACHINE",
                "formal_iterations": FORMAL_ITERATIONS,
                "formal_curriculum_max_level": 3,
                "checkpoint_limit": CHECKPOINT_LIMIT,
                "screening_seeds": list(SCREENING_SEEDS),
                "primary_video_seed": 42,
                "training_seed": 42,
                "clean_actor": True,
                "resume": False,
                "model_1999_allowed": False,
                "pushing_enabled": False,
                "planner_enabled": False,
                "falcon_migration": False,
            },
            "runner_interfaces": {
                "reachability": "scripts/stage2_isaac/run_s2_03t_reachability_smoke.py",
                "training": "scripts/stage2_isaac/train_s2_03t.py",
                "screening": "scripts/stage2_isaac/screen_s2_03t_redesign_checkpoints.py",
                "video": "scripts/stage2_isaac/record_s2_03t_redesign_video.py",
                "finalizer": "scripts/stage2/finalize_s2_03t_redesign_campaign.py",
            },
            "stage_records": [],
        }
        atomic_write_json(self.manifest_path, manifest)
        self.update_status(
            campaign_status="RUNNING",
            stage="REACHABILITY_SMOKE",
            primary_reason="CAMPAIGN_INITIALIZED",
            child_pid=None,
            command="NOT_STARTED",
        )
        self.record_history("CAMPAIGN_INITIALIZED", raw_rc=None, reason="STARTUP_CONTRACTS_PASSED")
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="campaign-heartbeat", daemon=True
        )
        self.heartbeat_thread.start()

    def update_status(self, **changes: Any) -> None:
        with self.state_lock:
            current = read_json(self.status_path) or {
                "schema_version": SCHEMA_VERSION,
                "campaign": "S2_03T_REDESIGN_TRAIN_SCREEN_AND_VIDEO_CAMPAIGN",
                "run_root": str(self.run_root),
                "tmux_session": self.args.tmux_session,
                "main_pid": os.getpid(),
                "started_at_utc": utc_timestamp(),
            }
            current.update(changes)
            current["updated_at_utc"] = utc_timestamp()
            current["last_checkpoint"] = latest_checkpoint(self.run_root)
            atomic_write_json(self.status_path, current)

    def record_history(self, event: str, raw_rc: int | None, reason: str, **extra: Any) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": utc_timestamp(),
            "elapsed_seconds": time.monotonic() - self.started_monotonic,
            "event": event,
            "stage": self.stage,
            "command": self.command,
            "raw_rc": raw_rc,
            "last_checkpoint": latest_checkpoint(self.run_root),
            "reason": reason,
            **extra,
        }
        append_jsonl(self.history_path, payload)

    def _gpu_snapshot(self) -> dict[str, Any]:
        try:
            output = command_output(
                (
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ),
                self.repository,
                timeout=5.0,
            )
            rows = []
            for line in output.splitlines():
                utilization, memory = (item.strip() for item in line.split(",", 1))
                rows.append(
                    {"utilization_percent": int(utilization), "memory_used_mib": int(memory)}
                )
            return {"available": True, "gpus": rows}
        except (OSError, ValueError, subprocess.SubprocessError):
            return {"available": False, "gpus": []}

    def _heartbeat_loop(self) -> None:
        while not self.stop_heartbeat.is_set():
            try:
                child_pid = None if self.child is None else self.child.pid
                log_path = self.run_root / "campaign.console.log"
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "timestamp_utc": utc_timestamp(),
                    "stage": self.stage,
                    "main_pid": os.getpid(),
                    "child_pid": child_pid,
                    "checkpoint": latest_checkpoint(self.run_root),
                    "last_log_offset": log_path.stat().st_size if log_path.is_file() else 0,
                    "gpu": self._gpu_snapshot(),
                    "elapsed_seconds": time.monotonic() - self.started_monotonic,
                    "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                }
                atomic_write_json(self.heartbeat_path, payload)
            except BaseException as exc:  # heartbeat failure is recorded, never hidden
                try:
                    append_jsonl(
                        self.history_path,
                        {
                            "timestamp_utc": utc_timestamp(),
                            "event": "HEARTBEAT_WRITE_FAILED",
                            "error": repr(exc),
                        },
                    )
                except BaseException:
                    pass
            self.stop_heartbeat.wait(HEARTBEAT_INTERVAL_SECONDS)

    def _update_manifest_stage(self, record: dict[str, Any]) -> None:
        manifest = read_json(self.manifest_path)
        if manifest is None:
            raise CampaignError("MANIFEST_MISSING_DURING_STAGE_UPDATE")
        records = manifest.setdefault("stage_records", [])
        if not isinstance(records, list):
            raise CampaignError("MANIFEST_STAGE_RECORDS_MALFORMED")
        records.append(record)
        manifest["updated_at_utc"] = utc_timestamp()
        atomic_write_json(self.manifest_path, manifest)

    def run_child(self, stage: str, command: Sequence[str], artifact: Path) -> StageResult:
        self._verify_execution_sources()
        self.stage = stage
        self.command = shlex.join(str(item) for item in command)
        started = utc_timestamp()
        self.update_status(
            campaign_status="RUNNING",
            stage=stage,
            primary_reason=f"{stage}_RUNNING",
            command=self.command,
            child_pid=None,
        )
        self.record_history(
            "STAGE_STARTED", raw_rc=None, reason=f"{stage}_STARTED", artifact=str(artifact)
        )
        self.child = subprocess.Popen(tuple(str(item) for item in command), cwd=self.repository)
        self.update_status(child_pid=self.child.pid)
        while True:
            raw_rc = self.child.poll()
            if raw_rc is not None:
                break
            if self.interrupted_signal is not None:
                self.child.terminate()
            time.sleep(2.0)
        self.child = None
        atomic_write_text(self.process_rc_root / f"{stage.lower()}.txt", f"{raw_rc}\n")
        payload = read_json(artifact)
        authoritative = payload is not None and payload.get("status") in {
            "PASS",
            "FAIL",
            "COMPLETE",
        }
        if raw_rc != 0 and not bool(
            (payload or {}).get("authoritative_complete_before_teardown", False)
        ):
            authoritative = False
            reason = f"{stage}_PROCESS_RC_{raw_rc}"
        elif payload is None:
            reason = f"{stage}_AUTHORITATIVE_JSON_MISSING"
        elif not authoritative:
            reason = str(payload.get("primary_reason") or f"{stage}_ARTIFACT_INVALID")
        else:
            reason = str(payload.get("primary_reason") or f"{stage}_{payload.get('status')}")
        record = {
            "stage": stage,
            "started_at_utc": started,
            "finished_at_utc": utc_timestamp(),
            "command": self.command,
            "raw_rc": raw_rc,
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact) if artifact.is_file() else None,
            "artifact_status": None if payload is None else payload.get("status"),
            "authoritative": authoritative,
            "reason": reason,
        }
        self.stage_records.append(record)
        self._update_manifest_stage(record)
        self.record_history(
            "STAGE_FINISHED", raw_rc=raw_rc, reason=reason, authoritative=authoritative
        )
        self.update_status(child_pid=None, stage=stage, primary_reason=reason, last_stage_rc=raw_rc)
        return StageResult(stage, raw_rc, artifact, payload, authoritative, reason)

    def _python(self, relative: str) -> list[str]:
        return [sys.executable, str(self.repository / relative)]

    def _common_reference_args(self) -> list[str]:
        return [
            "--reference",
            str(self.args.reference.resolve()),
            "--reference-sha256",
            self.args.reference_sha256,
            "--campaign-manifest",
            str(self.manifest_path),
        ]

    def reachability(self) -> StageResult:
        stage_root = self.run_root / "reachability_smoke"
        command = self._python("scripts/stage2_isaac/run_s2_03t_reachability_smoke.py")
        command += ["--run-root", str(stage_root), *self._common_reference_args()]
        command += ["--initial-gaps-m", *(f"{value:.3f}" for value in REACHABILITY_GAPS_M)]
        command += ["--record-diagnostic-on-failure", "--headless"]
        return self.run_child(
            "REACHABILITY_SMOKE", command, stage_root / "reachability_result.json"
        )

    def train(self, mode: str) -> StageResult:
        if mode == "pilot":
            stage, root, envs, iterations, interval, curriculum = (
                "PILOT_TRAINING",
                self.run_root / "pilot_training",
                PILOT_ENV_COUNT,
                PILOT_ITERATIONS,
                25,
                1,
            )
        elif mode == "formal":
            stage, root, envs, iterations, interval, curriculum = (
                "FORMAL_TRAINING",
                self.run_root / "formal_training",
                FORMAL_ENV_COUNT,
                FORMAL_ITERATIONS,
                100,
                3,
            )
        else:
            raise ValueError(mode)
        command = self._python("scripts/stage2_isaac/train_s2_03t.py")
        command += ["--run-root", str(root), *self._common_reference_args()]
        command += [
            "--mode",
            mode,
            "--num-envs",
            str(envs),
            "--max-iterations",
            str(iterations),
            "--save-interval",
            str(interval),
            "--curriculum-max-level",
            str(curriculum),
            "--seed",
            "42",
            "--headless",
        ]
        return self.run_child(stage, command, root / "training_result.json")

    def screening(self, training_root: Path, source: str) -> StageResult:
        stage_root = self.run_root / "checkpoint_screening"
        command = self._python("scripts/stage2_isaac/screen_s2_03t_redesign_checkpoints.py")
        command += [
            "--run-root",
            str(stage_root),
            "--training-root",
            str(training_root),
            "--training-source",
            source,
            *self._common_reference_args(),
            "--seeds",
            *(str(seed) for seed in SCREENING_SEEDS),
            "--max-checkpoints",
            str(CHECKPOINT_LIMIT),
            "--headless",
        ]
        return self.run_child("CHECKPOINT_SCREENING", command, stage_root / "screening_result.json")

    def video(self, screening_result: Path) -> StageResult:
        delivery = self.run_root / "video_delivery"
        command = self._python("scripts/stage2_isaac/record_s2_03t_redesign_video.py")
        command += [
            "--run-root",
            str(delivery),
            "--screening-result",
            str(screening_result),
            *self._common_reference_args(),
            "--primary-seed",
            "42",
            "--headless",
            "--enable_cameras",
        ]
        return self.run_child("TRAINED_POLICY_VIDEO", command, delivery / "video_result.json")

    def write_outcome(self, status: str, reason: str, policy_status: str) -> None:
        box_motion = box_motion_violation_evidence(self.run_root)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "campaign": "S2_03T_REDESIGN_TRAIN_SCREEN_AND_VIDEO_CAMPAIGN",
            "status": status,
            "primary_reason": reason,
            "new_policy_status": policy_status,
            "implementation_branch": EXPECTED_BRANCH,
            "implementation_commit": self.args.implementation_commit,
            "run_root": str(self.run_root),
            "tmux_session": self.args.tmux_session,
            "completed_at_utc": utc_timestamp(),
            "stage_records": self.stage_records,
            "safety_boundaries": {
                "second_instance_started": False,
                "old_checkpoint_resumed": False,
                "model_1999_used": False,
                "falcon_migration_started": False,
                "falcon_migration_decision": "NO",
                "box_push_commanded": False,
                "box_motion_violation_observed": box_motion["observed"],
                "box_pushed": box_motion["observed"],
                "box_motion_evidence": box_motion,
                "planner_started": False,
                "s2_04_started": False,
                "formal_s2_03_gates_unchanged": True,
            },
        }
        atomic_write_json(self.outcome_path, payload)
        self.update_status(
            campaign_status=status, stage="PRE_FINALIZER", primary_reason=reason, child_pid=None
        )

    def finalize(self) -> StageResult:
        command = self._python("scripts/stage2/finalize_s2_03t_redesign_campaign.py")
        command += [
            "--run-root",
            str(self.run_root),
            "--repository-root",
            str(self.repository),
            "--preservation-baseline",
            str(self.args.preservation_baseline.resolve()),
            "--implementation-commit",
            self.args.implementation_commit,
            "--branch",
            EXPECTED_BRANCH,
            "--remote",
            "origin",
        ]
        return self.run_child("FINALIZER", command, self.run_root / "finalizer_result.json")

    @staticmethod
    def _finalizer_failure_reason(result: StageResult) -> str:
        payload_reason = (result.payload or {}).get("primary_reason")
        if isinstance(payload_reason, str) and payload_reason:
            return payload_reason
        if result.reason:
            return str(result.reason)
        return "FINALIZER_ARTIFACT_INVALID"

    @staticmethod
    def _reachability_valid(result: StageResult) -> bool:
        payload = result.payload or {}
        cases = payload.get("cases")
        if (
            not result.authoritative
            or payload.get("diagnostic_valid") is not True
            or not isinstance(cases, list)
        ):
            return False
        try:
            actual_gaps = tuple(round(float(item["initial_gap_m"]), 6) for item in cases)
        except (KeyError, TypeError, ValueError):
            return False
        if actual_gaps != tuple(
            round(value, 6) for value in REACHABILITY_GAPS_M
        ) or not finite_tree(payload):
            return False
        if payload.get("status") == "FAIL":
            diagnostic = payload.get("diagnostic_replay")
            if not isinstance(diagnostic, Mapping) or diagnostic.get("status") != "PASS":
                return False
            for key in (
                "front_video",
                "side_video",
                "terminal_front_image",
                "terminal_side_image",
            ):
                record = diagnostic.get(key)
                if not isinstance(record, Mapping):
                    return False
                path = Path(str(record.get("path", "")))
                if (
                    not path.is_file()
                    or path.stat().st_size != record.get("size_bytes")
                    or sha256_file(path) != record.get("sha256")
                ):
                    return False
        return True

    @staticmethod
    def _training_valid(result: StageResult, iterations: int) -> bool:
        payload = result.payload or {}
        checkpoints = payload.get("checkpoints")
        return bool(
            result.raw_rc == 0
            and result.authoritative
            and payload.get("status") == "PASS"
            and payload.get("finite") is True
            and int(payload.get("completed_iterations", -1)) == iterations
            and isinstance(checkpoints, list)
            and checkpoints
            and finite_tree(payload)
        )

    @staticmethod
    def _screening_valid(result: StageResult) -> bool:
        payload = result.payload or {}
        count = payload.get("screening_episode_count")
        candidates = payload.get("candidate_count")
        try:
            expected = int(candidates) * len(SCREENING_SEEDS)
            count_value = int(count)
        except (TypeError, ValueError):
            return False
        return bool(
            result.authoritative
            and payload.get("status") in {"PASS", "FAIL"}
            and payload.get("diagnostic_valid") is True
            and 1 <= int(candidates) <= CHECKPOINT_LIMIT
            and count_value == expected
            and payload.get("screening_seeds") == list(SCREENING_SEEDS)
            and payload.get("new_policy_status") in {"PASS", "FAIL"}
            and finite_tree(payload)
        )

    @staticmethod
    def _video_valid(result: StageResult) -> bool:
        payload = result.payload or {}
        return bool(
            result.authoritative
            and payload.get("status") in {"PASS", "COMPLETE"}
            and payload.get("visualization_complete") is True
            and finite_tree(payload)
        )

    def execute(self) -> int:
        campaign_status = "INVALID"
        primary_reason = "CAMPAIGN_DID_NOT_REACH_A_TERMINAL_CLASSIFICATION"
        policy_status = "INVALID"
        try:
            self.initialize()
            reach = self.reachability()
            if not self._reachability_valid(reach):
                campaign_status, primary_reason = "INVALID", "REACHABILITY_EVIDENCE_INVALID"
            elif reach.payload and reach.payload.get("status") == "FAIL":
                campaign_status = "FAIL"
                primary_reason = str(
                    reach.payload.get("primary_reason") or "NEW_ACTION_REACHABILITY_SMOKE_FAILED"
                )
                policy_status = "INVALID"
            else:
                pilot = self.train("pilot")
                if not self._training_valid(pilot, PILOT_ITERATIONS):
                    campaign_status, primary_reason = "INVALID", "PILOT_TRAINING_EVIDENCE_INVALID"
                else:
                    self.stage = "PILOT_GATE"
                    self.command = "PURE_PYTHON_PILOT_GATE"
                    gate = evaluate_pilot_gate(pilot.payload or {})
                    atomic_write_json(self.run_root / "pilot_gate.json", gate)
                    self.record_history(
                        "PILOT_GATE_EVALUATED", raw_rc=0, reason=str(gate["primary_reason"])
                    )
                    self.update_status(stage="PILOT_GATE", primary_reason=gate["primary_reason"])
                    if gate["status"] == "INVALID":
                        campaign_status, primary_reason = "INVALID", str(gate["primary_reason"])
                    else:
                        training_root = self.run_root / "pilot_training"
                        source = "PILOT"
                        if gate["status"] == "PASS":
                            formal = self.train("formal")
                            if not self._training_valid(formal, FORMAL_ITERATIONS):
                                campaign_status, primary_reason = (
                                    "INVALID",
                                    "FORMAL_TRAINING_EVIDENCE_INVALID",
                                )
                                formal = None
                            else:
                                training_root = self.run_root / "formal_training"
                                source = "FORMAL"
                        else:
                            campaign_status, primary_reason = "FAIL", str(gate["primary_reason"])

                        if primary_reason != "FORMAL_TRAINING_EVIDENCE_INVALID":
                            screen = self.screening(training_root, source)
                            if not self._screening_valid(screen):
                                campaign_status, primary_reason, policy_status = (
                                    "INVALID",
                                    "CHECKPOINT_SCREENING_EVIDENCE_INVALID",
                                    "INVALID",
                                )
                            else:
                                policy_status = str(
                                    screen.payload.get(
                                        "new_policy_status", screen.payload.get("status")
                                    )
                                )
                                video = self.video(screen.artifact)
                                if not self._video_valid(video):
                                    campaign_status, primary_reason = (
                                        "INVALID",
                                        "VIDEO_EVIDENCE_INCOMPLETE",
                                    )
                                elif gate["status"] == "FAIL":
                                    campaign_status, primary_reason = (
                                        "FAIL",
                                        str(gate["primary_reason"]),
                                    )
                                elif policy_status == "PASS":
                                    campaign_status, primary_reason = (
                                        "PASS",
                                        "QUALIFIED_CONTACT_POLICY_AND_VIDEO_COMPLETE",
                                    )
                                elif policy_status == "FAIL":
                                    campaign_status = "FAIL"
                                    primary_reason = str(
                                        screen.payload.get("primary_reason")
                                        or "NO_QUALIFIED_CONTACT_POLICY"
                                    )
                                else:
                                    campaign_status, primary_reason = (
                                        "INVALID",
                                        "NEW_POLICY_STATUS_INVALID",
                                    )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                primary_reason = "CAMPAIGN_INTERRUPTED"
            elif isinstance(exc, CampaignError):
                primary_reason = str(exc)
            else:
                primary_reason = "ORCHESTRATOR_IMPLEMENTATION_EXCEPTION"
                atomic_write_json(
                    self.run_root / "orchestrator_exception.json",
                    {
                        "timestamp_utc": utc_timestamp(),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            campaign_status = "INVALID"
            policy_status = "INVALID"

        self.write_outcome(campaign_status, primary_reason, policy_status)
        scientific_status = campaign_status
        scientific_reason = primary_reason
        try:
            finalizer = self.finalize()
        except BaseException as finalizer_exc:
            finalizer_artifact = self.run_root / "finalizer_result.json"
            existing_finalizer = (
                read_json(finalizer_artifact) if finalizer_artifact.is_file() else None
            )
            existing_reason = (
                existing_finalizer.get("primary_reason")
                if isinstance(existing_finalizer, dict)
                and existing_finalizer.get("status") in {"FAIL", "INVALID"}
                else None
            )
            if isinstance(existing_reason, str) and existing_reason:
                finalizer_reason = existing_reason
            elif isinstance(finalizer_exc, KeyboardInterrupt):
                finalizer_reason = "CAMPAIGN_INTERRUPTED_DURING_FINALIZER"
            elif isinstance(finalizer_exc, CampaignError):
                finalizer_reason = str(finalizer_exc)
            else:
                finalizer_reason = "FINALIZER_ORCHESTRATION_EXCEPTION"
            exception_payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "INVALID",
                "primary_reason": finalizer_reason,
                "error": repr(finalizer_exc),
                "traceback": traceback.format_exc(),
                "result_commit": None,
                "all_code_commits_pushed": False,
                "all_remote_sha_verified": False,
                "authoritative_complete_before_teardown": False,
                "timestamp_utc": utc_timestamp(),
            }
            if not finalizer_artifact.exists():
                atomic_write_json(finalizer_artifact, exception_payload)
            else:
                atomic_write_json(
                    self.run_root / "finalizer_orchestration_exception.json",
                    exception_payload,
                )
            existing_finalizer = existing_finalizer or exception_payload
            self.update_status(
                campaign_status="INVALID",
                stage="DONE",
                primary_reason=finalizer_reason,
                child_pid=None,
                scientific_status_before_finalizer=scientific_status,
                scientific_reason_before_finalizer=scientific_reason,
                finalizer_status=existing_finalizer.get("status"),
                finalizer_primary_reason=existing_finalizer.get("primary_reason"),
                finalizer_rc=None,
                result_commit=existing_finalizer.get("result_commit"),
            )
            self.record_history(
                "CAMPAIGN_TERMINAL", raw_rc=None, reason=finalizer_reason, status="INVALID"
            )
            return 2
        if not finalizer.authoritative or (finalizer.payload or {}).get("status") != "PASS":
            campaign_status = "INVALID"
            primary_reason = self._finalizer_failure_reason(finalizer)
            self.update_status(
                campaign_status=campaign_status,
                stage="DONE",
                primary_reason=primary_reason,
                child_pid=None,
                scientific_status_before_finalizer=scientific_status,
                scientific_reason_before_finalizer=scientific_reason,
                finalizer_status=(finalizer.payload or {}).get("status"),
                finalizer_primary_reason=(finalizer.payload or {}).get("primary_reason"),
                finalizer_rc=finalizer.raw_rc,
                result_commit=(finalizer.payload or {}).get("result_commit"),
            )
            self.record_history("CAMPAIGN_TERMINAL", raw_rc=finalizer.raw_rc, reason=primary_reason)
            return 2

        self.update_status(
            campaign_status=campaign_status,
            stage="DONE",
            primary_reason=primary_reason,
            child_pid=None,
            finalizer_rc=finalizer.raw_rc,
            result_commit=(finalizer.payload or {}).get("result_commit"),
            all_code_commits_pushed=(finalizer.payload or {}).get("all_code_commits_pushed"),
            all_remote_sha_verified=(finalizer.payload or {}).get("all_remote_sha_verified"),
        )
        self.record_history(
            "CAMPAIGN_TERMINAL", raw_rc=0, reason=primary_reason, status=campaign_status
        )
        return 0 if campaign_status in {"PASS", "FAIL"} else 2

    def close(self) -> None:
        self.stop_heartbeat.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=5.0)
        heartbeat = read_json(self.heartbeat_path) or {}
        heartbeat.update(
            {
                "timestamp_utc": utc_timestamp(),
                "stage": "DONE",
                "main_pid": os.getpid(),
                "child_pid": None,
                "checkpoint": latest_checkpoint(self.run_root),
                "elapsed_seconds": time.monotonic() - self.started_monotonic,
                "heartbeat_stopped": True,
            }
        )
        atomic_write_json(self.heartbeat_path, heartbeat)

    def handle_signal(self, signum: int, _frame: object) -> None:
        self.interrupted_signal = signum
        self.failure_reason = f"SIGNAL_{signum}"
        self.update_status(
            campaign_status="INVALID",
            primary_reason=self.failure_reason,
            trap={
                "stage": self.stage,
                "rc": 128 + signum,
                "command": self.command,
                "timestamp_utc": utc_timestamp(),
                "last_checkpoint": latest_checkpoint(self.run_root),
                "reason": self.failure_reason,
            },
        )


def record_shell_exit(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    status_path = run_root / "status.json"
    status = read_json(status_path) or {"schema_version": SCHEMA_VERSION, "run_root": str(run_root)}
    prior_terminal = (
        status.get("campaign_status") in {"PASS", "FAIL", "INVALID"}
        and status.get("stage") == "DONE"
    )
    reason = "SHELL_EXIT_AFTER_TERMINAL_STATUS" if prior_terminal else "SHELL_EXIT"
    status["shell_exit"] = {
        "stage": status.get("stage", "SHELL_STARTUP"),
        "rc": args.exit_code,
        "command": args.command,
        "timestamp_utc": utc_timestamp(),
        "last_checkpoint": latest_checkpoint(run_root),
        "reason": reason,
    }
    if not prior_terminal and args.exit_code != 0:
        status.update(
            {
                "campaign_status": "INVALID",
                "primary_reason": "SHELL_OR_ORCHESTRATOR_EXIT_NONZERO",
                "updated_at_utc": utc_timestamp(),
            }
        )
    atomic_write_json(status_path, status)
    append_jsonl(
        run_root / "stage_history.jsonl",
        {"timestamp_utc": utc_timestamp(), "event": "SHELL_EXIT_TRAP", **status["shell_exit"]},
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--record-shell-exit", action="store_true")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--command", default="UNKNOWN")
    parser.add_argument("--tmux-session")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--preservation-baseline", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args(argv)
    if not args.record_shell_exit:
        required = {
            "tmux_session": args.tmux_session,
            "implementation_commit": args.implementation_commit,
            "repository_root": args.repository_root,
            "preservation_baseline": args.preservation_baseline,
            "reference": args.reference,
            "reference_sha256": args.reference_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"missing required campaign arguments: {', '.join(missing)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.record_shell_exit:
        return record_shell_exit(args)
    campaign = Campaign(args)
    signal.signal(signal.SIGINT, campaign.handle_signal)
    signal.signal(signal.SIGTERM, campaign.handle_signal)
    try:
        return campaign.execute()
    finally:
        campaign.close()


if __name__ == "__main__":
    raise SystemExit(main())
