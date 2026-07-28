#!/usr/bin/env python3
"""Record and package deterministic post-training S2-03T visual evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from g1_access_push.stage2.s2_03t_redesign_contract import (
    validate_campaign_execution_manifest,
)

PRIMARY_SEED = 42
FIXED_SEEDS = (42, 43, 44)
REQUIRED_EXECUTION_SOURCES = (
    "scripts/stage2_isaac/record_s2_03t_redesign_video.py",
    "scripts/stage2_isaac/evaluate_s2_03t_checkpoint.py",
    "scripts/stage2_isaac/s2_03t_visual_recorder.py",
    "src/g1_access_push/sim/stage2/s2_03t_actions.py",
    "src/g1_access_push/sim/stage2/s2_03t_env.py",
    "src/g1_access_push/sim/stage2/s2_03t_env_cfg.py",
    "src/g1_access_push/sim/stage2/s2_03t_mdp.py",
    "src/g1_access_push/stage2/s2_03t_redesign_contract.py",
)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"DELIVERY_FILE_MISSING_OR_EMPTY:{path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return payload


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def best_checkpoint(screening: Mapping[str, Any]) -> dict[str, Any]:
    value = screening.get("best_checkpoint")
    if not isinstance(value, Mapping):
        raise ValueError("BEST_CHECKPOINT_OBJECT_MISSING")
    required = {"path", "sha256", "iteration", "episodes"}
    if not required.issubset(value):
        # The compact top-level object may omit episodes; recover the exact full
        # candidate by path without changing the authoritative selection.
        results = screening.get("checkpoint_results")
        if not isinstance(results, list):
            raise ValueError("BEST_CHECKPOINT_EPISODES_MISSING")
        matches = [
            item
            for item in results
            if isinstance(item, Mapping) and item.get("path") == value.get("path")
        ]
        if len(matches) != 1:
            raise ValueError("BEST_CHECKPOINT_FULL_RECORD_NOT_UNIQUE")
        return dict(matches[0])
    return dict(value)


def episode_for_seed(candidate: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    episodes = candidate.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("BEST_CHECKPOINT_EPISODES_MISSING")
    matches = [
        item for item in episodes if isinstance(item, Mapping) and int(item.get("seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"SCREENING_SEED_RECORD_NOT_UNIQUE:{seed}")
    return matches[0]


def copy_episode_delivery(episode_root: Path, delivery_root: Path, role: str) -> dict[str, Any]:
    role_lower = role.lower()
    copies = {
        f"{role_lower}_front.mp4": episode_root / "video_three_quarter.mp4",
        f"{role_lower}_side.mp4": episode_root / "video_side.mp4",
        f"{role_lower}_visual_result.json": episode_root / "visual_evidence.json",
    }
    for destination_name, source in copies.items():
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"VISUAL_SOURCE_MISSING:{source}")
        shutil.copyfile(source, delivery_root / destination_name)
    keyframes = {}
    for name in ("initial", "precontact", "minimum_gap", "terminal"):
        for suffix in ("", "_side"):
            source = episode_root / f"frame_{name}{suffix}.png"
            destination = delivery_root / f"{role_lower}_frame_{name}{suffix}.png"
            if not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"VISUAL_KEYFRAME_MISSING:{source}")
            shutil.copyfile(source, destination)
            keyframes[destination.name] = file_record(destination)
    return {
        "front": file_record(delivery_root / f"{role_lower}_front.mp4"),
        "side": file_record(delivery_root / f"{role_lower}_side.mp4"),
        "visual_result": file_record(delivery_root / f"{role_lower}_visual_result.json"),
        "keyframes": keyframes,
    }


def run_visual_episode(
    *,
    role: str,
    seed: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_iteration: int,
    reference: Path,
    reference_sha256: str,
    campaign_manifest: Path,
    run_root: Path,
    evaluator: Path,
) -> dict[str, Any]:
    episode_root = run_root / f"_{role.lower()}_episode"
    validate_campaign_execution_manifest(
        campaign_manifest,
        stage_run_root=episode_root,
        reference_path=reference,
        reference_sha256=reference_sha256,
        required_source_relatives=REQUIRED_EXECUTION_SOURCES,
    )
    command = [
        sys.executable,
        str(evaluator),
        "--run-root",
        str(episode_root),
        "--reference",
        str(reference),
        "--reference-sha256",
        reference_sha256,
        "--campaign-manifest",
        str(campaign_manifest),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        checkpoint_sha256,
        "--checkpoint-iteration",
        str(checkpoint_iteration),
        "--controller-id",
        "actor",
        "--mode",
        "screening",
        "--development-seed",
        str(seed),
        "--visual-evidence",
        "--visual-evidence-id",
        role,
        "--visual-frame-stride",
        "2",
        "--headless",
        "--enable_cameras",
    ]
    log_path = run_root / f"{role.lower()}.console.log"
    with log_path.open("wb") as log:
        completed = subprocess.run(command, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT)
    result = (
        read_json(episode_root / "result.json") if (episode_root / "result.json").is_file() else {}
    )
    visualization = (
        read_json(episode_root / "visualization_status.json")
        if (episode_root / "visualization_status.json").is_file()
        else {}
    )
    visual_evidence = (
        read_json(episode_root / "visual_evidence.json")
        if (episode_root / "visual_evidence.json").is_file()
        else {}
    )
    if result.get("status") not in {"PASS", "FAIL"} or not finite_tree(result):
        raise ValueError(f"{role}_SCIENTIFIC_RESULT_INVALID")
    if visualization.get("visualization_status") != "PASS":
        raise ValueError(
            f"{role}_VISUALIZATION_INVALID:{visualization.get('visualization_primary_reason')}"
        )
    if (
        completed.returncode != 0
        and result.get("authoritative_complete_before_teardown") is not True
    ):
        raise ValueError(f"{role}_EVALUATOR_RC_{completed.returncode}")
    return {
        "role": role,
        "seed": seed,
        "episode_root": str(episode_root),
        "process_rc": completed.returncode,
        "scientific_status": result["status"],
        "scientific_primary_reason": result["primary_reason"],
        "contact_occurred": bool(result.get("contact_occurred", False)),
        "bilateral_contact_achieved": bool(result.get("bilateral_contact_achieved", False)),
        "observed_frames": visual_evidence.get("observed_frames"),
        "scientific_trace_observed_frames": result.get("observed_frames"),
        "result_json": str(episode_root / "result.json"),
        "visual_evidence_json": str(episode_root / "visual_evidence.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--screening-result", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--primary-seed", type=int, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    result_path = run_root / "video_result.json"
    status_path = run_root / "video_status.json"
    atomic_write_json(status_path, {"status": "STARTING", "timestamp_utc": utc_timestamp()})
    try:
        if args.primary_seed != PRIMARY_SEED:
            raise ValueError("PRIMARY_VIDEO_SEED_MUST_BE_42")
        if not args.headless or not args.enable_cameras:
            raise ValueError("VIDEO_REQUIRES_HEADLESS_ENABLE_CAMERAS")
        reference = args.reference.resolve()
        if not reference.is_file() or sha256_file(reference) != args.reference_sha256:
            raise ValueError("PRECONTACT_REFERENCE_SHA_MISMATCH")
        manifest_path = args.campaign_manifest.resolve()
        manifest = validate_campaign_execution_manifest(
            manifest_path,
            stage_run_root=run_root,
            reference_path=reference,
            reference_sha256=args.reference_sha256,
            required_source_relatives=REQUIRED_EXECUTION_SOURCES,
        )
        frozen = manifest.get("frozen_execution", {})
        if frozen.get("action_dim") != 2 or frozen.get("screening_seeds") != list(FIXED_SEEDS):
            raise ValueError("CAMPAIGN_VISUAL_CONTRACT_MISMATCH")
        screening = read_json(args.screening_result.resolve())
        if (
            screening.get("status") not in {"PASS", "FAIL"}
            or screening.get("diagnostic_valid") is not True
            or screening.get("screening_seeds") != list(FIXED_SEEDS)
        ):
            raise ValueError("SCREENING_RESULT_NOT_AUTHORITATIVE")
        candidate = best_checkpoint(screening)
        checkpoint = Path(str(candidate["path"])).resolve()
        checkpoint_sha = str(candidate["sha256"])
        checkpoint_iteration = int(candidate["iteration"])
        if checkpoint.name == "model_1999.pt":
            raise ValueError("MODEL_1999_FORBIDDEN")
        if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_sha:
            raise ValueError("BEST_CHECKPOINT_SHA_MISMATCH")
        primary_screen = episode_for_seed(candidate, PRIMARY_SEED)
        evaluator = Path(__file__).with_name("evaluate_s2_03t_checkpoint.py")
        primary = run_visual_episode(
            role="PRIMARY",
            seed=PRIMARY_SEED,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_iteration=checkpoint_iteration,
            reference=reference,
            reference_sha256=args.reference_sha256,
            campaign_manifest=manifest_path,
            run_root=run_root,
            evaluator=evaluator,
        )
        primary["screening_episode"] = dict(primary_screen)
        primary["delivery"] = copy_episode_delivery(
            Path(primary["episode_root"]), run_root, "PRIMARY"
        )
        atomic_write_json(
            status_path,
            {"status": "RUNNING", "phase": "PRIMARY_COMPLETE", "timestamp_utc": utc_timestamp()},
        )

        supplemental = None
        if not bool(primary_screen.get("contact_occurred", False)):
            contact_seeds = [
                seed
                for seed in FIXED_SEEDS
                if seed != PRIMARY_SEED
                and bool(episode_for_seed(candidate, seed).get("contact_occurred", False))
            ]
            if contact_seeds:
                supplemental_seed = min(contact_seeds)
                supplemental = run_visual_episode(
                    role="SUPPLEMENTAL",
                    seed=supplemental_seed,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha,
                    checkpoint_iteration=checkpoint_iteration,
                    reference=reference,
                    reference_sha256=args.reference_sha256,
                    campaign_manifest=manifest_path,
                    run_root=run_root,
                    evaluator=evaluator,
                )
                supplemental["screening_episode"] = dict(
                    episode_for_seed(candidate, supplemental_seed)
                )
                supplemental["delivery"] = copy_episode_delivery(
                    Path(supplemental["episode_root"]), run_root, "SUPPLEMENTAL"
                )

        readme = run_root / "README.txt"
        readme.write_text(
            "S2-03T redesigned-policy deterministic visual evidence.\n"
            "PRIMARY is always seed 42. SUPPLEMENTAL, when present, is the smallest fixed seed "
            "with any left-or-right palm screening contact because PRIMARY had none.\n"
            "Videos are diagnostic presentation; result.json/trace/contact sensors remain scientific authority.\n"
            "No pushing, planner, retraining, resume, or safety-gate override was used.\n",
            encoding="utf-8",
        )
        delivery_names = [
            "primary_front.mp4",
            "primary_side.mp4",
            "primary_visual_result.json",
            "README.txt",
        ]
        if supplemental is not None:
            delivery_names.extend(
                (
                    "supplemental_front.mp4",
                    "supplemental_side.mp4",
                    "supplemental_visual_result.json",
                )
            )
        video_manifest = {
            "schema_version": 1,
            "package": "S2_03T_TRAINED_POLICY_VIDEO_DELIVERY",
            "status": "PASS",
            "primary_reason": "PRIMARY_AND_REQUIRED_SUPPLEMENTAL_VISUAL_EVIDENCE_READY",
            "generated_at_utc": utc_timestamp(),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "iteration": checkpoint_iteration,
            },
            "primary": primary,
            "supplemental": supplemental,
            "files": {name: file_record(run_root / name) for name in delivery_names},
            "execution": {
                "sequential": True,
                "parallel_isaac_instances": False,
                "num_envs": 1,
                "episode_count_per_replay": 1,
                "deterministic": True,
                "auto_reset_accumulation": False,
                "box_push_commanded": False,
                "planner_started": False,
                "formal_s2_03_gates_unchanged": True,
                "supplemental_contact_semantics": "ANY_LEFT_OR_RIGHT_PALM_CONTACT",
            },
        }
        atomic_write_json(run_root / "video_manifest.json", video_manifest)

        archive = run_root.parent / "S2_03T_TRAINED_POLICY_VIDEO_DELIVERY.tar.gz"
        archive_names = [*delivery_names, "video_manifest.json"]
        with tarfile.open(archive, "w:gz") as stream:
            for name in archive_names:
                stream.add(run_root / name, arcname=name, recursive=False)
        archive_record = file_record(archive)
        result = {
            "schema_version": 1,
            "stage": "S2-03T_REDESIGN_TRAINED_POLICY_VIDEO",
            "status": "COMPLETE",
            "primary_reason": "TRAINED_POLICY_VIDEO_DELIVERY_COMPLETE",
            "visualization_complete": True,
            "authoritative_complete_before_teardown": True,
            "primary_seed": PRIMARY_SEED,
            "primary_video_recorded": True,
            "primary": primary,
            "supplemental_video_recorded": supplemental is not None,
            "supplemental": supplemental,
            "video_manifest": file_record(run_root / "video_manifest.json"),
            "archive": archive_record,
            "large_artifacts_committed_to_git": False,
            "completed_at_utc": utc_timestamp(),
        }
        atomic_write_json(result_path, result)
        atomic_write_json(status_path, {"status": "COMPLETE", "timestamp_utc": utc_timestamp()})
        return 0
    except BaseException as exc:
        atomic_write_json(
            result_path,
            {
                "schema_version": 1,
                "stage": "S2-03T_REDESIGN_TRAINED_POLICY_VIDEO",
                "status": "INVALID",
                "primary_reason": "VIDEO_IMPLEMENTATION_OR_EVIDENCE_EXCEPTION",
                "visualization_complete": False,
                "authoritative_complete_before_teardown": True,
                "error": repr(exc),
                "completed_at_utc": utc_timestamp(),
            },
        )
        atomic_write_json(
            status_path, {"status": "INVALID", "error": repr(exc), "timestamp_utc": utc_timestamp()}
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
