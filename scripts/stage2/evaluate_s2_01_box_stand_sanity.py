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


def implementation_exception_evidence(run: Path) -> dict | None:
    """Resolve implementation failure before checking whether a trace exists."""
    marker = run / "implementation_exception.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return {"source": marker.name, **payload}
    traceback_logs = []
    for name in ("stderr.log", "stdout.log", "console.log"):
        path = run / name
        if path.is_file() and "Traceback (most recent call last)" in path.read_text(encoding="utf-8", errors="replace"):
            traceback_logs.append(name)
    raw_path = run / "runner_status.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.is_file() else {}
    runner_rc_path = run / "process_rc/runner.txt"
    runner_rc = int(runner_rc_path.read_text().strip()) if runner_rc_path.is_file() else None
    empty_pre_environment = raw.get("environment_created") is False and int(raw.get("observed_frames", 0)) == 0
    if traceback_logs or raw.get("primary_reason") == "IMPLEMENTATION_EXCEPTION" or empty_pre_environment:
        return {
            "source": "DERIVED_PROCESS_EVIDENCE",
            "traceback_logs": traceback_logs,
            "runner_rc": runner_rc,
            "environment_created": raw.get("environment_created"),
            "observed_frames": raw.get("observed_frames"),
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root
    implementation = implementation_exception_evidence(run)
    if implementation is not None:
        result = {
            "schema_version": 1, "stage": "S2-01", "status": "INVALID",
            "primary_reason": "IMPLEMENTATION_EXCEPTION",
            "implementation_exception_evidence": implementation,
        }
        write_json(run / "result.json", result)
        return 2
    trace = run / "trace.jsonl"
    if not trace.is_file():
        result = {"schema_version": 1, "stage": "S2-01", "status": "INVALID", "primary_reason": "MISSING_TRACE"}
        write_json(run / "result.json", result)
        return 2
    try:
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
        audits = {
            name: json.loads((run / name).read_text(encoding="utf-8"))
            for name in REQUIRED_AUDIT_FIELDS
            if (run / name).is_file()
        }
        runner_rc_path = run / "process_rc/runner.txt"
        runner_rc = int(runner_rc_path.read_text().strip()) if runner_rc_path.is_file() else 1
        raw = json.loads((run / "runner_status.json").read_text()) if (run / "runner_status.json").is_file() else {}
        decision = classify_evidence(
            load_config(args.config), records, audits, runner_rc=runner_rc,
            final_image_present=(run / "final.png").is_file(),
            multiple_isaac_processes=bool(raw.get("multiple_isaac_processes", False)),
        )
        result = {
            "schema_version": 1, "stage": "S2-01",
            "task": "BOX_SPAWN_NO_CONTACT_STAND_SANITY",
            **decision,
            "development_baseline_id": "S2_LIGHT_BOX_DEVELOPMENT_BASELINE_V1",
            "episode_count": 1, "expected_frames": 3000,
            "observed_frames": len(records), "reset_count": 0,
            "trace_complete": len(records) == 3000,
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
            "schema_version": 1, "stage": "S2-01", "status": "INVALID",
            "primary_reason": "EVALUATOR_DID_NOT_COMPLETE", "error": repr(exc),
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
