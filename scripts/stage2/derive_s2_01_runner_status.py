#!/usr/bin/env python3
"""Derive authoritative S2-01 runner status without importing Isaac or Kit."""

from __future__ import annotations

import argparse
from pathlib import Path

from g1_access_push.stage2.s2_01_process import persist_effective_runner_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-rc", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, default=3000)
    args = parser.parse_args()
    result = persist_effective_runner_status(args.run_root, args.raw_rc, args.expected_frames)
    print(f"RUNNER_RAW_RC={result['runner_raw_rc']}")
    print(f"RUNNER_EFFECTIVE_RC={result['runner_effective_rc']}")
    print(f"RUNNER_EFFECTIVE_STATUS={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
