"""Pure source and manifest contracts for S2-03T visual evidence."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/stage2_isaac/evaluate_s2_03t_checkpoint.py"
RECORDER = ROOT / "scripts/stage2_isaac/s2_03t_visual_recorder.py"
CAMERA_CFG = ROOT / "src/g1_access_push/sim/stage2/s2_03t_visual_cfg.py"
LAUNCHER = ROOT / "scripts/stage2_isaac/run_s2_03t_visual_evidence_once.sh"
SUMMARY = ROOT / "scripts/stage2/derive_s2_03t_visual_summary.py"
SCREENING = ROOT / "reports/stage2/s2_03t_screening_summary.json"
CHECKPOINT = Path("/root/autodl-tmp/robotics/runs/g1_access_push/stage2/s2_03t_formal_clean_contactfix_20260728_113403/model_500.pt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_visual_sources_are_syntactically_valid_and_separate_science_status() -> None:
    for path in (EVALUATOR, RECORDER, CAMERA_CFG, SUMMARY):
        ast.parse(path.read_text(encoding="utf-8"))
    recorder = RECORDER.read_text(encoding="utf-8")
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    assert '"visualization_status": "PASS"' in recorder
    assert '"scientific_result_status": terminal_state' in recorder
    assert '"terminal_capture": "PRE_AUTO_RESET_PHYSICS_FRAME"' in recorder
    assert "self.observe(terminal_record" not in recorder
    assert "visual_terminal_callback" in evaluator
    assert "ORIGINAL_CONTROLLER_MUST_NOT_LOAD_ACTOR_CHECKPOINT" in evaluator
    assert not any(token in recorder.splitlines() for token in ("true", "false", "null"))


def test_original_controller_flattens_two_palm_poses_for_pose_error() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    assert "palms.data.target_pos_source.reshape(-1, 3)" in evaluator
    assert "palms.data.target_quat_source.reshape(-1, 4)" in evaluator
    assert "desired_pos.reshape(-1, 3)" in evaluator
    assert "target_quat.reshape(-1, 4)" in evaluator
    assert "pos_error.reshape(env.num_envs, 2, 3)" in evaluator
    assert "orientation_error.reshape(env.num_envs, 2, 3)" in evaluator


def test_two_fixed_cameras_and_required_overlay_fields_are_frozen() -> None:
    cfg = CAMERA_CFG.read_text(encoding="utf-8")
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    recorder = RECORDER.read_text(encoding="utf-8")
    assert "torch.tensor([[-1.6, -2.2, 1.40]]" in evaluator
    assert "torch.tensor([[0.45, 0.0, 0.70]]" in evaluator
    assert "VISUAL_WIDTH = 640" in cfg and "VISUAL_HEIGHT = 480" in cfg
    assert "set_world_poses_from_view" in evaluator
    for field in ("frame=", "left_gap_m=", "right_gap_m=", "left_contact=", "right_contact=", "left_force_n=", "right_force_n=", "root_tilt_deg="):
        assert field in recorder
    for image in ("frame_initial.png", "frame_precontact.png", "frame_minimum_gap.png", "frame_terminal.png"):
        assert image in recorder


def test_launcher_runs_the_three_episodes_in_order_without_parallelism() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    positions = [source.index(label) for label in ("CLEAN_UNTRAINED_ACTOR", "BEST_GAP_CHECKPOINT", "ORIGINAL_S2_03_CONTROLLER")]
    assert positions == sorted(positions)
    assert "run_episode CLEAN_UNTRAINED_ACTOR" in source
    assert "run_episode BEST_GAP_CHECKPOINT" in source
    assert "run_episode ORIGINAL_S2_03_CONTROLLER" in source
    assert " &" not in source and "parallel" not in source.lower()
    assert "--visual-evidence" in source and "--visual-frame-stride 2" in source


def test_summary_generator_is_pure_and_emits_small_manifest(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    labels = ("CLEAN_UNTRAINED_ACTOR", "BEST_GAP_CHECKPOINT", "ORIGINAL_S2_03_CONTROLLER")
    for label in labels:
        episode = package / label
        episode.mkdir()
        (episode / "result.json").write_text(json.dumps({"status": "FAIL", "primary_reason": "BILATERAL_CONTACT_TIMEOUT"}), encoding="utf-8")
        (episode / "visualization_status.json").write_text(json.dumps({"visualization_status": "PASS", "visualization_primary_reason": "OK"}), encoding="utf-8")
        (episode / "visual_evidence.json").write_text(json.dumps({
            "observed_frames": 10,
            "seed": 42,
            "reference": {"sha256": "ref"},
            "minimum_gap": {"left_m": 0.02, "right_m": 0.01, "mean_m": 0.015, "mean_frame": 9},
            "contact": {"left_seen": False, "right_seen": False},
            "action_and_motion": {"maximum_abs_commanded_residual_rad": 0.05},
            "videos": {}, "required_images": {},
        }), encoding="utf-8")
        (episode / "trace.jsonl").write_text("{}\n" * 10, encoding="utf-8")
        (episode / "video_three_quarter.mp4").write_bytes(b"video")
        (episode / "video_side.mp4").write_bytes(b"video")
        (episode / "frame_initial.png").write_bytes(b"png")
    command = [
        sys.executable, str(SUMMARY), "--package-root", str(package),
        "--screening-summary", str(SCREENING), "--best-checkpoint", str(CHECKPOINT),
        "--best-checkpoint-sha256", sha256(CHECKPOINT), "--best-checkpoint-iteration", "500",
        "--development-seed", "42", "--launcher-rc-clean", "0", "--launcher-rc-best", "0", "--launcher-rc-original", "0",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    assert completed.returncode == 1  # intentionally incomplete image set => package INVALID
    summary = json.loads((package / "visual_evidence_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((package / "visual_evidence_sha256_manifest.json").read_text(encoding="utf-8"))
    assert summary["visualization_package_status"] == "INVALID"
    assert len(summary["episodes"]) == 3
    assert manifest["files"]
