#!/usr/bin/env python3
"""Pure evaluator for one S2-02 formal no-contact episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_access_push.stage2.s2_02_contract import classify_formal, load_config, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        records = [
            json.loads(line)
            for line in (args.run_root / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        decision = classify_formal(config, records, final_image_present=(args.run_root / "final.png").is_file())
        selection = config["selection"]
        result = {
            "schema_version": 1,
            "stage": "S2-02",
            "task": "PRECONTACT_GEOMETRY_AND_BIMANUAL_TARGET_AUDIT",
            **decision,
            "episode_count": 1,
            "expected_frames": int(config["formal"]["expected_frames"]),
            "observed_frames": len(records),
            "reset_count": max((int(record.get("post_initial_reset_count", 0)) for record in records), default=0),
            "selected_candidate": selection,
            "final_image_present": (args.run_root / "final.png").is_file(),
            "contact_occurred": any(float(record.get("robot_box_contact_force_n", 0.0)) > 0.0 for record in records),
            "box_pushed": False,
            "attach_run": False,
        }
        write_json(args.run_root / "evaluation_summary.json", result)
        write_json(args.run_root / "result.json", result)
        return 0 if result["status"] in {"PASS", "FAIL"} else 2
    except Exception as exc:
        write_json(args.run_root / "result.json", {
            "schema_version": 1,
            "stage": "S2-02",
            "status": "INVALID",
            "primary_reason": "IMPLEMENTATION_EXCEPTION",
            "error": repr(exc),
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
