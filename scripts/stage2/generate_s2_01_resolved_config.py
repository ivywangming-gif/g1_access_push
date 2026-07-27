#!/usr/bin/env python3
"""Generate deterministic S2-01 resolved config and parameter provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from g1_access_push.stage2.s2_01_contract import resolved_config, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage2/s2_01_box_stand_sanity.yaml")
    parser.add_argument("--resolved", type=Path, default=ROOT / "reports/stage2/s2_01_resolved_config.json")
    parser.add_argument("--provenance", type=Path, default=ROOT / "reports/stage2/s2_01_parameter_provenance.json")
    args = parser.parse_args()
    resolved = resolved_config(args.config)
    write_json(args.resolved, resolved)
    cfg = resolved["resolved"]
    standing = Path(cfg["certification"]["standing_result_path"])
    collision = Path(cfg["certification"]["collision_backend_path"])
    provenance = {
        "schema_version": 1,
        "baseline": cfg["baseline"],
        "previous_invalid_evidence": "/tmp/g1_s2_01_20260727_124751/parameter_source_audit.json",
        "previous_status_retained": "INVALID",
        "previous_primary_reason": "BOX_PHYSICS_PARAMETER_SOURCE_AMBIGUOUS",
        "previous_secondary_reason": "INITIAL_CLEARANCE_CONTRACT_UNRESOLVED",
        "current_baseline_resolution": "RESEARCH_LEAD_EXPLICIT_DEVELOPMENT_DECISION_2026_07_27",
        "standing_source": {
            "path": str(standing), "sha256": sha256_file(standing),
            "expected_sha256": cfg["certification"]["standing_result_sha256"],
        },
        "collision_backend": {
            "path": str(collision), "sha256": sha256_file(collision),
            "expected_sha256": cfg["certification"]["collision_backend_sha256"],
            "passed": json.loads(collision.read_text())["passed"],
        },
        "controller_checkpoint": {
            "sha256": cfg["robot"]["controller_checkpoint_sha256"],
            "forbidden_legacy_checkpoint_used": False,
        },
        "box_actor_overrides": {
            name: {"actor_override": "NOT_AUTHORED", "effective_source": "PHYSX_SCENE_OR_ENGINE_DEFAULT"}
            for name in cfg["simulation"]["actor_properties_not_authored"]
        },
        "contact_model": cfg["contact_model"],
        "palm_box_material": cfg["materials"]["palm_box"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    if provenance["standing_source"]["sha256"] != provenance["standing_source"]["expected_sha256"]:
        raise RuntimeError("standing result SHA mismatch")
    if provenance["collision_backend"]["sha256"] != provenance["collision_backend"]["expected_sha256"]:
        raise RuntimeError("collision backend SHA mismatch")
    write_json(args.provenance, provenance)


if __name__ == "__main__":
    main()
