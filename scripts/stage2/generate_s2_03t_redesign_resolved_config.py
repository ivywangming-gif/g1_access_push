#!/usr/bin/env python3
"""Generate the deterministic S2-03T redesign resolved configuration."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from g1_access_push.stage2.s2_03t_redesign_contract import build_resolved_config, canonical_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/stage2/s2_03t_redesign_resolved_config.json"


def atomic_write_text(path: Path, text: str) -> None:
    """Write, fsync, and atomically replace one text artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=Path,
        default=ROOT / "configs/stage2/s2_03t_contact_action_contract.yaml",
    )
    parser.add_argument(
        "--reward",
        type=Path,
        default=ROOT / "configs/stage2/s2_03t_contact_curriculum_reward_contract.yaml",
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=ROOT / "configs/stage2/s2_03t_contact_curriculum_contract.yaml",
    )
    parser.add_argument(
        "--attach", type=Path, default=ROOT / "configs/stage2/s2_03_attach_only.yaml"
    )
    parser.add_argument(
        "--legacy-training",
        type=Path,
        default=ROOT / "configs/stage2/s2_03t_contact_training_contract.yaml",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that --output already contains canonical content.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_resolved_config(
        action_path=args.action,
        reward_path=args.reward,
        curriculum_path=args.curriculum,
        attach_path=args.attach,
        legacy_training_path=args.legacy_training,
    )
    expected = canonical_json(payload)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            print(f"S2_03T_REDESIGN_RESOLVED_CONFIG=STALE output={args.output}")
            return 1
        print(f"S2_03T_REDESIGN_RESOLVED_CONFIG=PASS output={args.output}")
        return 0
    atomic_write_text(args.output, expected)
    print(f"S2_03T_REDESIGN_RESOLVED_CONFIG=PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
