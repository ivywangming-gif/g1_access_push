"""Run every S1-06 motion primitive in an independent episode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = (
    PROJECT_ROOT
    / "scripts/stage1/run_s1_06_base_motion_hand_hold.py"
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=Path,
    default=(
        PROJECT_ROOT
        / "configs/stage1/s1_06_base_motion_hand_hold.yaml"
    ),
)
parser.add_argument(
    "--output_json",
    type=Path,
    required=True,
)
parser.add_argument(
    "--device",
    default="cuda:0",
)
parser.add_argument(
    "--headless",
    action="store_true",
)
args = parser.parse_args()


def run_case(
    *,
    suite_config: dict,
    segment: dict,
    case_index: int,
    output_root: Path,
) -> dict:
    name = str(segment["name"])
    case_dir = output_root / "cases" / (
        f"{case_index:02d}_{name}"
    )

    if case_dir.exists():
        raise FileExistsError(case_dir)

    case_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    case_config = {
        "test_id": (
            f"{suite_config['test_id']}_{name}"
        ),
        "seed": int(suite_config["seed"]),
        "warmup_steps": int(
            suite_config["warmup_steps"]
        ),
        "transient_ignore_steps": int(
            suite_config["transient_ignore_steps"]
        ),
        "segments": [segment],
        "acceptance": suite_config["acceptance"],
    }

    config_path = case_dir / "config.yaml"
    result_path = case_dir / "result.json"
    console_path = case_dir / "console.log"

    config_path.write_text(
        yaml.safe_dump(
            case_config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(WORKER),
        "--config",
        str(config_path),
        "--output_json",
        str(result_path),
        "--device",
        args.device,
    ]

    if args.headless:
        command.append("--headless")

    timed_out = False

    with console_path.open(
        "w",
        encoding="utf-8",
    ) as console:
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=console,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=float(
                    suite_config["case_timeout_s"]
                ),
            )
            raw_rc = int(
                completed.returncode
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            raw_rc = 124

    (
        case_dir / "raw_process_rc.txt"
    ).write_text(
        f"{raw_rc}\n",
        encoding="utf-8",
    )

    result = None
    parse_error = None

    if result_path.is_file():
        try:
            result = json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as error:
            parse_error = repr(error)

    checks = (
        result.get("checks", {})
        if isinstance(result, dict)
        else {}
    )

    failed_checks = sorted(
        check_name
        for check_name, passed
        in checks.items()
        if passed is not True
    )

    passed = bool(
        raw_rc == 0
        and not timed_out
        and isinstance(result, dict)
        and result.get("passed") is True
        and not failed_checks
    )

    return {
        "name": name,
        "raw_process_rc": raw_rc,
        "timed_out": timed_out,
        "parse_error": parse_error,
        "passed": passed,
        "failed_checks": failed_checks,
        "case_dir": str(case_dir),
        "result": result,
    }


def main() -> int:
    if not WORKER.is_file():
        raise FileNotFoundError(WORKER)

    config = yaml.safe_load(
        args.config.read_text(
            encoding="utf-8"
        )
    )

    if config.get("episode_mode") != "isolated":
        raise ValueError(
            "S1-06 suite requires "
            "episode_mode='isolated'."
        )

    segments = config.get("segments")

    if not isinstance(segments, list):
        raise ValueError(
            "segments must be a list."
        )

    if not segments:
        raise ValueError(
            "segments must not be empty."
        )

    names = [
        str(segment["name"])
        for segment in segments
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            f"Duplicate segment names: {names}"
        )

    output_root = (
        args.output_json.parent
    )

    if (
        args.output_json.exists()
        or (output_root / "cases").exists()
    ):
        raise FileExistsError(
            "Refusing to overwrite an "
            "existing S1-06 suite result."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = {}

    for index, segment in enumerate(
        segments,
        start=1,
    ):
        name = str(segment["name"])

        print(
            f"RUNNING_S1_06_CASE={name}",
            flush=True,
        )

        case = run_case(
            suite_config=config,
            segment=segment,
            case_index=index,
            output_root=output_root,
        )

        cases[name] = case

        print(
            f"case={name} "
            f"rc={case['raw_process_rc']} "
            f"passed={case['passed']} "
            f"failed={case['failed_checks']}",
            flush=True,
        )

    passed = all(
        case["passed"]
        for case in cases.values()
    )

    result = {
        "passed": passed,
        "protocol": {
            "episode_mode": "isolated",
            "seed": int(config["seed"]),
            "warmup_steps_per_episode": int(
                config["warmup_steps"]
            ),
            "transient_ignore_steps": int(
                config["transient_ignore_steps"]
            ),
            "case_count": len(cases),
        },
        "acceptance": config["acceptance"],
        "cases": cases,
    }

    args.output_json.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "STAGE1_S1_06_ISOLATED_SUITE: "
        + ("PASS" if passed else "FAIL"),
        flush=True,
    )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
