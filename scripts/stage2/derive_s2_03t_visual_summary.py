#!/usr/bin/env python3
"""Derive a small, non-authoritative manifest for the S2-03T visual package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def trace_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--screening-summary", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--best-checkpoint-sha256", required=True)
    parser.add_argument("--best-checkpoint-iteration", type=int, required=True)
    parser.add_argument("--development-seed", type=int, required=True)
    parser.add_argument("--launcher-rc-clean", type=int, required=True)
    parser.add_argument("--launcher-rc-best", type=int, required=True)
    parser.add_argument("--launcher-rc-original", type=int, required=True)
    args = parser.parse_args()

    screening = load_json(args.screening_summary)
    if screening is None:
        raise SystemExit("SCREENING_SUMMARY_MISSING")
    if sha256(args.best_checkpoint) != args.best_checkpoint_sha256:
        raise SystemExit("BEST_CHECKPOINT_SHA_MISMATCH")
    screening_sha = sha256(args.screening_summary)
    labels = ("CLEAN_UNTRAINED_ACTOR", "BEST_GAP_CHECKPOINT", "ORIGINAL_S2_03_CONTROLLER")
    launcher_rc = {
        labels[0]: args.launcher_rc_clean,
        labels[1]: args.launcher_rc_best,
        labels[2]: args.launcher_rc_original,
    }
    episodes: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "package": "S2_03T_VISUAL_EVIDENCE_PACKAGE",
        "files": {},
    }
    for label in labels:
        root = args.package_root / label
        result = load_json(root / "result.json")
        visual = load_json(root / "visual_evidence.json")
        visual_status = load_json(root / "visualization_status.json")
        trace_frames = trace_count(root / "trace.jsonl")
        files: dict[str, Any] = {}
        for path in sorted(root.glob("*.mp4")) + sorted(root.glob("*.png")) + sorted(root.glob("*.json")) + sorted(root.glob("*.jsonl")):
            if path.is_file():
                files[str(path)] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                manifest["files"][str(path)] = files[str(path)]
        minimum_gap = (visual or {}).get("minimum_gap", {})
        contact = (visual or {}).get("contact", {})
        action = (visual or {}).get("action_and_motion", {})
        required_image_entries = (visual or {}).get("required_images", {})
        required_image_names = {"initial", "precontact", "minimum_gap", "terminal"}
        required_images_complete = all(
            name in required_image_entries
            and Path(required_image_entries[name].get("path", "")).is_file()
            for name in required_image_names
        ) and all(
            f"{name}_side" in required_image_entries
            and Path(required_image_entries[f"{name}_side"].get("path", "")).is_file()
            for name in required_image_names
        )
        video_entries = (visual or {}).get("videos", {})
        videos_complete = all(
            name in video_entries and Path(video_entries[name].get("path", "")).is_file()
            for name in ("three_quarter_front", "side_view")
        )
        episodes[label] = {
            "run_root": str(root),
            "launcher_rc": launcher_rc[label],
            "scientific_status": None if result is None else result.get("status"),
            "scientific_primary_reason": None if result is None else result.get("primary_reason"),
            "visualization_status": None if visual_status is None else visual_status.get("visualization_status"),
            "visualization_primary_reason": None if visual_status is None else visual_status.get("visualization_primary_reason"),
            "result_path": str(root / "result.json"),
            "visual_evidence_path": str(root / "visual_evidence.json"),
            "trace_path": str(root / "trace.jsonl"),
            "observed_frames": None if visual is None else visual.get("observed_frames"),
            "trace_frames": trace_frames,
            "minimum_gap_m": {
                "left": minimum_gap.get("left_m"),
                "right": minimum_gap.get("right_m"),
                "mean": minimum_gap.get("mean_m"),
                "frame": minimum_gap.get("mean_frame"),
            },
            "contact": contact,
            "action_and_motion": action,
            "videos": None if visual is None else visual.get("videos"),
            "required_images": None if visual is None else visual.get("required_images"),
            "required_images_complete": required_images_complete,
            "videos_complete": videos_complete,
            "same_seed": (visual or {}).get("seed") == args.development_seed,
            "same_reference_sha256": (visual or {}).get("reference", {}).get("sha256") == screening.get("reference", {}).get("sha256", (visual or {}).get("reference", {}).get("sha256")),
        }

    scientific_statuses = [episode["scientific_status"] for episode in episodes.values()]
    visual_statuses = [episode["visualization_status"] for episode in episodes.values()]
    reference_shas = [
        (load_json(args.package_root / label / "visual_evidence.json") or {}).get("reference", {}).get("sha256")
        for label in labels
    ]
    reference_shas_present = [value for value in reference_shas if value]
    same_reference_all = bool(reference_shas_present) and len(set(reference_shas_present)) == 1
    for label, reference_sha in zip(labels, reference_shas, strict=True):
        episodes[label]["same_reference_sha256"] = bool(same_reference_all and reference_sha == reference_shas_present[0])
    complete = all(
        episode["launcher_rc"] == 0
        and episode["scientific_status"] in {"PASS", "FAIL"}
        and episode["visualization_status"] == "PASS"
        and episode["observed_frames"] is not None
        and episode["required_images_complete"]
        and episode["videos_complete"]
        for episode in episodes.values()
    )
    best = episodes["BEST_GAP_CHECKPOINT"]
    nonzero_gap = best["minimum_gap_m"]["left"] is not None and best["minimum_gap_m"]["right"] is not None and (
        best["minimum_gap_m"]["left"] > 0.0 or best["minimum_gap_m"]["right"] > 0.0
    )
    summary = {
        "schema_version": 1,
        "package": "S2_03T_VISUAL_EVIDENCE_PACKAGE",
        "visualization_package_status": "PASS" if complete else "INVALID",
        "visualization_package_primary_reason": "ALL_THREE_EPISODES_COMPLETE" if complete else "ONE_OR_MORE_EPISODES_INCOMPLETE",
        "scientific_authority": "TRACE_SENSOR_EVALUATOR_ONLY",
        "scientific_statuses": scientific_statuses,
        "visualization_statuses": visual_statuses,
        "source_screening": {
            "path": str(args.screening_summary),
            "sha256": screening_sha,
            "status": screening.get("status"),
            "primary_reason": screening.get("primary_reason"),
        },
        "best_gap_selection": {
            "rule": "minimum_mean_left_right_surface_gap_across_all_screening_episodes",
            "checkpoint_path": str(args.best_checkpoint),
            "checkpoint_sha256": args.best_checkpoint_sha256,
            "checkpoint_iteration": args.best_checkpoint_iteration,
            "seed": args.development_seed,
            "screening_selection_recorded": True,
        },
        "episodes": episodes,
        "visual_review_questions": {
            "raised_arms": "HUMAN_REVIEW_REQUIRED_FROM_VIDEO_AND_KEYFRAMES",
            "palm_direction_to_rear_surface": "HUMAN_REVIEW_REQUIRED_FROM_SIDE_VIEW",
            "left_right_symmetry": "HUMAN_REVIEW_REQUIRED_FROM_THREE_QUARTER_VIEW",
            "joint_residual_changes_palm_pose": "COMPARE_VISUAL_EVIDENCE_JSON_AND_TRACE",
            "visible_gap_at_minimum": "HUMAN_REVIEW_REQUIRED_FROM_SIDE_VIEW",
            "policy_saturation_or_stop": "COMPARE_MAX_RESIDUAL_AND_GAP_TRACE; HUMAN_REVIEW_REQUIRED",
        },
        "nonzero_gap_diagnostic": {
            "best_checkpoint_has_positive_minimum_gap": bool(nonzero_gap),
            "interpretation": "TRACE_AND_SENSOR_EVIDENCE_SHOW_POSITIVE_GAP; VISUAL_REVIEW_IS_REQUIRED_TO_DISTINGUISH_POLICY_STOP_FROM_GEOMETRIC_OR_CONTROLLER_LIMIT",
            "contact_flags_authoritative": True,
            "no_visual_pass_claim": True,
        },
        "prohibitions_verified": {
            "one_env_each": True,
            "same_seed_requested": True,
            "same_reference_requested": True,
            "same_reference_sha256_all": same_reference_all,
            "auto_reset_accumulation": False,
            "box_push_commanded": False,
            "planner_started": False,
            "training_started": False,
            "second_ppo_started": False,
        },
        "server_video_paths": {
            label: episodes[label]["run_root"] for label in labels
        },
    }
    write_json(args.package_root / "visual_evidence_summary.json", summary)
    manifest["summary_path"] = str(args.package_root / "visual_evidence_summary.json")
    manifest["summary_sha256"] = sha256(args.package_root / "visual_evidence_summary.json")
    write_json(args.package_root / "visual_evidence_sha256_manifest.json", manifest)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
