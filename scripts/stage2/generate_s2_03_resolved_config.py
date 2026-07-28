#!/usr/bin/env python3
"""Generate deterministic S2-03 resolved config and parameter provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

from g1_access_push.stage2.s2_03_contract import resolved_config, write_json

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage2/s2_03_attach_only.yaml")
    parser.add_argument("--resolved", type=Path, default=ROOT / "reports/stage2/s2_03_resolved_config.json")
    parser.add_argument("--provenance", type=Path, default=ROOT / "reports/stage2/s2_03_parameter_provenance.json")
    args = parser.parse_args()
    resolved, provenance = resolved_config(
        args.config,
        ROOT / "reports/stage2/s2_01_runtime_audit_summary.json",
        ROOT / "reports/stage2/s2_02_formal_result.json",
        ROOT / "configs/stage2/s2_01_box_stand_sanity.yaml",
    )
    write_json(args.resolved, resolved)
    write_json(args.provenance, provenance)


if __name__ == "__main__":
    main()
