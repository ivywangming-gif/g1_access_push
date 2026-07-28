#!/usr/bin/env python3
"""Persist effective S2-03 runner RC from authoritative JSON and trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-rc", type=int, required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    args = parser.parse_args()
    runner = read_json(args.run_root / "runner_status.json")
    outcome = read_json(args.run_root / "fsm_outcome.json")
    marker = (args.run_root / "implementation_exception.json").is_file()
    trace_path = args.run_root / "trace.jsonl"
    trace_frames = sum(1 for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()) if trace_path.is_file() else None
    reason = None
    if marker:
        reason = "IMPLEMENTATION_EXCEPTION"
    elif runner is None or runner.get("status") != "COMPLETE":
        reason = "MISSING_OR_INCOMPLETE_RUNNER_STATUS"
    elif runner.get("environment_created") is not True:
        reason = "ENVIRONMENT_NOT_CREATED"
    elif trace_frames is None or int(runner.get("observed_frames", -1)) != trace_frames:
        reason = "INCOMPLETE_TRACE"
    elif args.mode == "preflight" and trace_frames != 3:
        reason = "INCOMPLETE_PREFLIGHT_TRACE"
    elif args.mode == "formal" and (outcome is None or outcome.get("terminal_state") not in {"PASS", "FAIL"}):
        reason = "MISSING_TERMINAL_FSM_OUTCOME"
    effective_rc = 0 if reason is None else 1
    result = {
        "schema_version": 1, "mode": args.mode, "runner_raw_rc": args.raw_rc,
        "runner_effective_rc": effective_rc, "status": "COMPLETE" if reason is None else "INVALID",
        "primary_reason": reason, "observed_frames": None if runner is None else runner.get("observed_frames"),
        "trace_frames": trace_frames, "environment_created": None if runner is None else runner.get("environment_created"),
        "implementation_exception_present": marker,
        "raw_rc_nonzero_after_authoritative_completion": bool(args.raw_rc != 0 and reason is None),
    }
    process = args.run_root / "process_rc"
    process.mkdir(parents=True, exist_ok=True)
    for name, value in (("runner_raw.txt", args.raw_rc), ("runner_effective.txt", effective_rc), ("runner.txt", effective_rc)):
        (process / name).write_text(f"{value}\n", encoding="utf-8")
    temporary = args.run_root / "runner_effective_status.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.run_root / "runner_effective_status.json")
    return 0 if effective_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
