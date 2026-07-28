#!/usr/bin/env python3
"""Pure evaluator for the single S2-03 attach-only formal episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_access_push.stage2.s2_03_contract import classify_formal, load_config, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        records = [json.loads(line) for line in (args.run_root / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        outcome = json.loads((args.run_root / "fsm_outcome.json").read_text(encoding="utf-8"))
        final_image = (args.run_root / "final.png").is_file()
        decision = classify_formal(config, records, outcome, final_image_present=final_image)
        hold = [record for record in records if record.get("fsm_state") == "ATTACHED_HOLD"]
        result = {
            "schema_version": 1, "stage": "S2-03", "task": "BILATERAL_ATTACH_ONLY", **decision,
            "episode_count": 1, "observed_frames": len(records), "reset_count": max((int(r.get("post_initial_reset_count", 0)) for r in records), default=0),
            "final_image_present": final_image, "fsm_outcome": outcome,
            "contact_occurred": any(r.get("left_contact") or r.get("right_contact") for r in records),
            "bilateral_contact_achieved": outcome.get("bilateral_contact_onset_frame") is not None,
            "attached_hold_observed_frames": len(hold), "box_push_commanded": False,
            "maximum_metrics": {
                key: max((float(r[key]) for r in records), default=0.0)
                for key in ("left_force_n", "right_force_n", "left_impulse_ns", "right_impulse_ns", "combined_impulse_ns", "contact_force_peak_n", "contact_force_rate_nps", "box_translation_m", "box_yaw_change_rad", "box_linear_speed_mps", "box_angular_speed_radps", "root_tilt_deg", "arm_torque_ratio_max")
            },
            "minimum_root_height_m": min((float(r["root_height_m"]) for r in records), default=None),
            "minimum_arm_joint_limit_margin_rad": min((float(r["minimum_arm_joint_limit_margin_rad"]) for r in records), default=None),
            "forbidden_contact_links": sorted({link for r in records for link in r.get("forbidden_contact_links", [])}),
        }
        write_json(args.run_root / "evaluation_summary.json", result)
        write_json(args.run_root / "result.json", result)
        return 0 if result["status"] in {"PASS", "FAIL"} else 2
    except Exception as exc:
        write_json(args.run_root / "result.json", {
            "schema_version": 1, "stage": "S2-03", "status": "INVALID",
            "primary_reason": "IMPLEMENTATION_EXCEPTION", "error": repr(exc),
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
