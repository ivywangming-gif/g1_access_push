#!/usr/bin/env python3
"""Generate deterministic S2-02 resolved configuration."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from g1_access_push.stage2.s2_02_contract import resolved_config, write_json

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage2/s2_02_precontact_audit.yaml")
    parser.add_argument("--resolved", type=Path, default=ROOT / "reports/stage2/s2_02_resolved_config.json")
    args = parser.parse_args()
    result = resolved_config(args.config)
    result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(args.resolved, result)


if __name__ == "__main__":
    main()
