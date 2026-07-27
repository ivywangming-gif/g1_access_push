#!/usr/bin/env python3
"""Generate a non-runnable, deterministic S2-00 resolved contract."""

from __future__ import annotations

import argparse

from g1_access_push.stage2.resolved_config import resolve_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="development YAML")
    parser.add_argument("output", help="resolved JSON")
    args = parser.parse_args()
    result = resolve_config(args.config, output_path=args.output)
    print(f"RESOLVED_CONFIG={args.output}")
    print(f"RUNNABLE={str(result['runnable']).lower()}")
    print(f"UNRESOLVED_COUNT={len(result['unresolved_parameters'])}")
    print(f"CONTRACT_DIGEST_SHA256={result['contract_digest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
