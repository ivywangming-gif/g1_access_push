#!/usr/bin/env python3
"""Persist S2-02 effective runner RC independently of Kit teardown."""

from __future__ import annotations

import argparse
from pathlib import Path

from g1_access_push.stage2.s2_01_process import persist_effective_runner_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-rc", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    args = parser.parse_args()
    result = persist_effective_runner_status(args.run_root, args.raw_rc, args.expected_frames)
    return 0 if result["runner_effective_rc"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
