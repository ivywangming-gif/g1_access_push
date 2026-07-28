#!/usr/bin/env python3
"""Pure evaluator for one completed S2-01 trace; never imports simulator modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_access_push.stage2.s2_01_contract import (
    REQUIRED_AUDIT_FIELDS,
    classify_evidence,
    load_config,
    write_json,
)


def _json_or_empty(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _rc(run: Path) -> int | None:
    for name in ("runner_effective.txt", "runner.txt"):
        path = run / "process_rc" / name
        if path.is_file():
            return int(path.read_text(encoding="utf-8").strip())
    return None


def _trace_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def implementation_exception_evidence(run: Path, expected_frames: int = 3000) -> dict | None:
    """Return incomplete-run evidence while allowing certified complete runs to ignore log noise."""
    marker = run / "implementation_exception.json"
    if marker.is_file():
        payload = _json_or_empty(marker)
        message = payload.get("exception_message")
        reason = message if message in {
            "CONTACT_SENSOR_INITIALIZATION_FAILED",
            "CONTACT_SENSOR_AUDIT_IMPLEMENTATION_ERROR",
        } else "IMPLEMENTATION_EXCEPTION"
        return {**payload, "source": marker.name, "primary_reason": reason}

    runner = _json_or_empty(run / "runner_status.json")
    effective = _json_or_empty(run / "runner_effective_status.json")
    effective_rc = _rc(run)
    trace_frames = _trace_count(run / "trace.jsonl")
    complete = (
        runner.get("status") == "COMPLETE"
        and runner.get("environment_created") is True
        and int(runner.get("observed_frames", -1)) == expected_frames
        and trace_frames == expected_frames
        and effective_rc == 0
    )
    if complete:
        return None

    traceback_logs = []
    for name in ("stderr.log", "stdout.log", "console.log"):
        path = run / name
        if path.is_file() and "Traceback (most recent call last)" in path.read_text(encoding="utf-8", errors="replace"):
            traceback_logs.append(name)
    reason = effective.get("primary_reason")
    if not reason:
        if runner.get("status") not in (None, "COMPLETE"):
            reason = runner.get("primary_reason") or "RUNNER_NOT_COMPLETE"
        elif runner.get("environment_created") is False:
            reason = "ENVIRONMENT_NOT_CREATED"
        elif trace_frames is None:
            reason = "MISSING_TRACE"
        elif trace_frames != expected_frames:
            reason = "INCOMPLETE_TRACE"
        elif effective_rc not in (None, 0):
            reason = "IMPLEMENTATION_EXCEPTION"
        elif traceback_logs:
            reason = "IMPLEMENTATION_EXCEPTION"
    if reason is None:
        return None
    return {
        "source": "DERIVED_PROCESS_EVIDENCE",
        "primary_reason": reason,
        "traceback_logs": traceback_logs,
        "runner_effective_rc": effective_rc,
        "runner_status": runner.get("status"),
        "environment_created": runner.get("environment_created"),
        "observed_frames": runner.get("observed_frames"),
        "trace_frames": trace_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root
    config = load_config(args.config)
    expected_frames = int(config["evaluation"]["expected_frames"])
    incomplete = implementation_exception_evidence(run, expected_frames)
    if incomplete is not None:
        result = {
            "schema_version": 1,
            "stage": "S2-01",
            "status": "INVALID",
            "primary_reason": incomplete["primary_reason"],
            "runner_incomplete_evidence": incomplete,
        }
        write_json(run / "result.json", result)
        return 2
    trace = run / "trace.jsonl"
    try:
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
        audits = {
            name: json.loads((run / name).read_text(encoding="utf-8"))
            for name in REQUIRED_AUDIT_FIELDS
            if (run / name).is_file()
        }
        runner_rc = _rc(run)
        raw = _json_or_empty(run / "runner_status.json")
        decision = classify_evidence(
            config,
            records,
            audits,
            runner_rc=1 if runner_rc is None else runner_rc,
            final_image_present=(run / "final.png").is_file(),
            multiple_isaac_processes=bool(raw.get("multiple_isaac_processes", False)),
        )
        result = {
            "schema_version": 1,
            "stage": "S2-01",
            "task": "BOX_SPAWN_NO_CONTACT_STAND_SANITY",
            **decision,
            "development_baseline_id": "S2_LIGHT_BOX_DEVELOPMENT_BASELINE_V1",
            "episode_count": 1,
            "expected_frames": expected_frames,
            "observed_frames": len(records),
            "reset_count": 0,
            "trace_complete": len(records) == expected_frames,
            "final_image_present": (run / "final.png").is_file(),
            "runtime_audits": sorted(audits),
            "robot_stability": "PASS" if not any(reason.startswith("ROBOT_") or reason in {"NONFINITE", "AUTO_RESET_DETECTED"} for reason in decision["all_reasons"]) else "FAIL",
            "box_stillness": "PASS" if not any(reason.startswith("OBJECT_") or reason == "GROUND_SUPPORT_CONTACT_MISSING" for reason in decision["all_reasons"]) else "FAIL",
            "evidence": {"ATTACH_NOT_RUN": True, "BOX_NOT_PUSHED": True, "PLANNER_NOT_STARTED": True, "S2_02_NOT_STARTED": True},
        }
        write_json(run / "evaluation_summary.json", result)
        write_json(run / "result.json", result)
        return 0
    except Exception as exc:
        write_json(run / "result.json", {
            "schema_version": 1,
            "stage": "S2-01",
            "status": "INVALID",
            "primary_reason": "EVALUATOR_DID_NOT_COMPLETE",
            "error": repr(exc),
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
