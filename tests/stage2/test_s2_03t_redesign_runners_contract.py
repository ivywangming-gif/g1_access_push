"""Runtime-free contracts for redesign reachability, screening, and video runners."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISAAC = ROOT / "scripts/stage2_isaac"
REACH = ISAAC / "run_s2_03t_reachability_smoke.py"
EVALUATE = ISAAC / "evaluate_s2_03t_checkpoint.py"
SCREEN = ISAAC / "screen_s2_03t_redesign_checkpoints.py"
VIDEO = ISAAC / "record_s2_03t_redesign_video.py"
RECORDER = ISAAC / "s2_03t_visual_recorder.py"
CAMPAIGN = ROOT / "scripts/stage2/s2_03t_redesign_campaign.py"


def load_pure_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = load_pure_module("s2_03t_redesign_screen_static", SCREEN)
video = load_pure_module("s2_03t_redesign_video_static", VIDEO)


def test_all_runner_sources_parse_without_launching_isaac() -> None:
    for path in (REACH, EVALUATE, SCREEN, VIDEO, RECORDER):
        ast.parse(path.read_text(encoding="utf-8"))


def test_campaign_cli_and_authoritative_artifact_names_match_exactly() -> None:
    source = CAMPAIGN.read_text(encoding="utf-8")
    assert "scripts/stage2_isaac/run_s2_03t_reachability_smoke.py" in source
    assert "scripts/stage2_isaac/screen_s2_03t_redesign_checkpoints.py" in source
    assert "scripts/stage2_isaac/record_s2_03t_redesign_video.py" in source
    assert 'stage_root / "reachability_result.json"' in source
    assert 'stage_root / "screening_result.json"' in source
    assert 'delivery / "video_result.json"' in source
    reach = REACH.read_text(encoding="utf-8")
    screening = SCREEN.read_text(encoding="utf-8")
    visual = VIDEO.read_text(encoding="utf-8")
    for token in ("--run-root", "--reference", "--reference-sha256", "--campaign-manifest"):
        assert token in reach and token in screening and token in visual
    for token in ("--initial-gaps-m", "--record-diagnostic-on-failure"):
        assert token in reach
    for token in ("--training-root", "--training-source", "--seeds", "--max-checkpoints"):
        assert token in screening
    assert "--screening-result" in visual and "--primary-seed" in visual


def test_all_runtime_runners_revalidate_manifest_and_pass_it_to_evaluator() -> None:
    reach = REACH.read_text(encoding="utf-8")
    evaluator = EVALUATE.read_text(encoding="utf-8")
    screening = SCREEN.read_text(encoding="utf-8")
    visual = VIDEO.read_text(encoding="utf-8")
    for source in (reach, evaluator, screening, visual):
        assert "validate_campaign_execution_manifest(" in source
    assert reach.index("validate_campaign_execution_manifest(") < reach.index(
        "simulation_app = AppLauncher"
    )
    assert evaluator.index("validate_campaign_execution_manifest(") < evaluator.index(
        "simulation_app = AppLauncher"
    )
    assert screening.count("validate_campaign_execution_manifest(") >= 2
    assert visual.count("validate_campaign_execution_manifest(") >= 2
    assert 'parser.add_argument("--campaign-manifest", type=Path, required=True)' in evaluator
    assert screening.count('"--campaign-manifest"') >= 2
    assert visual.count('"--campaign-manifest"') >= 2


def test_reachability_is_five_gap_zero_action_no_actor_no_ppo() -> None:
    source = REACH.read_text(encoding="utf-8")
    assert "EXPECTED_GAPS_M = (0.0, 0.005, 0.010, 0.030, 0.060)" in source
    assert '"episode_count": 5' in source
    assert '"actor_loaded": False' in source
    assert '"ppo_started": False' in source
    assert '"box_push_commanded": False' in source
    assert "torch.zeros((1, 2)" in source
    assert "OnPolicyRunner" not in source
    assert "runner.load" not in source
    assert "env.pop_terminal_snapshot(0)" in source
    assert "INITIAL_GAP_PLACEMENT_TOLERANCE_M" in source
    assert 'case["measured_initial_gap_within_placement_tolerance"]' in source
    assert "capture_diagnostic_terminal" in source
    assert '"terminal_frame_included_in_videos": True' in source
    step_index = source.index("observation, reward, done, _ = wrapped.step(zero_action)")
    finite_index = source.index("torch.isfinite(reward).all()", step_index)
    done_index = source.index("if bool(done[0]):", step_index)
    assert step_index < finite_index < done_index
    assert '"no_auto_reset_accumulation": True' in source
    assert '"authoritative_complete_before_teardown": True' in source


def test_evaluator_has_true_camera_free_2d_single_episode_path() -> None:
    source = EVALUATE.read_text(encoding="utf-8")
    assert "ACTOR_OBSERVATION_DIM = 135" in source
    assert "CRITIC_OBSERVATION_DIM = 175" in source
    assert "ACTOR_ACTION_DIM = 2" in source
    assert "S203TContactEnvCfg()" in source
    assert 'render_mode="rgb_array" if args.enable_cameras else None' in source
    assert "if bool(done[0]):" in source
    assert "terminal_snapshot = env.pop_terminal_snapshot(0)" in source
    assert "break" in source
    assert '"reset_count": 0' in source
    assert '"camera_free_screening"' in source
    assert '"formal_s2_03_scientific_gates_unchanged": True' in source
    assert '"terminal_trace_capture": "PRE_AUTO_RESET_PHYSICS_FRAME"' in source
    assert '"post_done_steps_executed": 0' in source
    assert '"mean_bilateral_force_imbalance_n"' in source
    assert "classify_formal(" in source
    assert "runner.load(str(checkpoint_path), load_optimizer=False)" in source
    assert "model_1999.pt" in source


def test_checkpoint_selection_covers_non_reward_contact_metrics() -> None:
    checkpoints = [
        {
            "path": f"/tmp/model_{iteration}.pt",
            "sha256": str(iteration),
            "iteration_field": iteration,
        }
        for iteration in (0, 100, 200, 300, 400, 500, 599)
    ]
    curve = []
    for iteration in range(600):
        curve.append(
            {
                "iteration": iteration,
                "verify_fraction": 0.9 if iteration == 101 else 0.1,
                "hold_success_fraction": 0.8 if iteration == 201 else 0.0,
                "contact_retention_fraction": 0.7 if iteration == 301 else 0.0,
                "minimum_safe_gap_m": 0.001 if iteration == 401 else 0.02,
                "hard_safety_violation_fraction": 0.0,
                "pushing_fraction": 0.0,
                "force_band_occupancy": 0.5,
                "episode_return": 10_000.0 if iteration == 500 else 0.0,
            }
        )
    selected = screen.select_checkpoints(
        {"checkpoints": checkpoints, "curve": curve},
        6,
    )
    iterations = {screen.checkpoint_iteration(item) for item in selected}
    assert 599 in iterations  # final
    assert {100, 200, 300, 400}.issubset(iterations)
    assert len(selected) <= 6
    final = next(item for item in selected if screen.checkpoint_iteration(item) == 599)
    assert final["post_update_behavior_metric_missing"] is True
    assert "episode_return" not in SCREEN.read_text(encoding="utf-8")


def test_minimum_safe_gap_selection_and_checkpoint_dimensions_are_explicit() -> None:
    checkpoints = [
        {
            "path": f"/tmp/model_{iteration}.pt",
            "sha256": str(iteration),
            "iteration_field": iteration,
        }
        for iteration in (100, 200)
    ]
    curve = [
        {
            "iteration": 101,
            "mean_surface_gap_m": 0.010,
            "hard_safety_violation_fraction": 0.0,
            "pushing_fraction": 0.0,
        },
        {
            "iteration": 201,
            "mean_surface_gap_m": 0.001,
            "hard_safety_violation_fraction": 1.0,
            "pushing_fraction": 0.0,
        },
    ]
    assert screen.safe_gap_score(checkpoints[0], curve) > screen.safe_gap_score(
        checkpoints[1], curve
    )
    valid = {
        "actor_input_dim": 135,
        "critic_input_dim": 175,
        "actor_output_dim": 2,
    }
    screen.validate_checkpoint_policy_contract(valid)
    try:
        screen.validate_checkpoint_policy_contract({**valid, "actor_output_dim": 14})
    except ValueError as exc:
        assert "CHECKPOINT_POLICY_CONTRACT_MISMATCH" in str(exc)
    else:
        raise AssertionError("legacy 14D checkpoint must be rejected")


def test_screening_rank_is_safety_first_and_three_seed_qualified() -> None:
    safe_episodes = [
        {
            "hard_safety_reasons": [],
            "pushing_reasons": [],
            "bilateral_verify_completed": True,
            "safe_hold_completed": True,
            "contact_retention_steps": 100,
            "minimum_surface_gap_m": [0.0, 0.0],
            "maximum_force_n": [2.0, 2.0],
        }
        for _ in range(3)
    ]
    unsafe_episodes = [dict(item, hard_safety_reasons=["force_peak"]) for item in safe_episodes]
    safe = {"iteration": 100, "episodes": safe_episodes}
    unsafe = {"iteration": 500, "episodes": unsafe_episodes}
    assert screen.checkpoint_rank(safe) > screen.checkpoint_rank(unsafe)
    source = SCREEN.read_text(encoding="utf-8")
    assert "qualified_count == len(EXPECTED_SEEDS)" in source
    assert "EXPECTED_SEEDS = (42, 43, 44)" in source
    assert "subprocess.run" in source and "subprocess.Popen" not in source
    assert '"camera_enabled": False' in source
    assert 'result.get("camera_enabled") is False' in source
    assert 'result.get("camera_free_screening") is True' in source
    assert 'legacy_packaging.get("primary_reason") == "MISSING_FINAL_IMAGE"' in source
    assert 'result.get("evaluation_mode") == "SCREENING"' in source
    assert 'result.get("controller_id") == "actor"' in source


def test_video_primary_is_seed42_and_supplemental_cannot_hide_it() -> None:
    source = VIDEO.read_text(encoding="utf-8")
    assert video.PRIMARY_SEED == 42
    assert video.FIXED_SEEDS == (42, 43, 44)
    assert "primary = run_visual_episode(" in source
    assert "supplemental_seed = min(contact_seeds)" in source
    assert "if not bool(primary_screen.get" in source
    assert 'primary_screen.get("contact_occurred", False)' in source
    assert 'episode_for_seed(candidate, seed).get("contact_occurred", False)' in source
    assert "subprocess.run" in source and "subprocess.Popen" not in source
    assert '"parallel_isaac_instances": False' in source
    assert '"visualization_complete": True' in source
    assert "S2_03T_TRAINED_POLICY_VIDEO_DELIVERY.tar.gz" in source
    assert "primary_front.mp4" in source and "primary_side.mp4" in source
    assert "supplemental_front.mp4" in source and "supplemental_side.mp4" in source


def test_delivery_archive_excludes_raw_trace_checkpoint_and_png() -> None:
    source = VIDEO.read_text(encoding="utf-8")
    members_block = source[
        source.index("        delivery_names = [") : source.index("        video_manifest = {")
    ]
    assert "trace.jsonl" not in members_block
    assert ".pt" not in members_block
    assert ".png" not in members_block
    assert 'archive_names = [*delivery_names, "video_manifest.json"]' in source
    assert "keyframes" in source  # delivered on server, intentionally outside tar list


def test_visual_recorder_tracks_2d_correction_and_14d_joint_effect() -> None:
    source = RECORDER.read_text(encoding="utf-8")
    assert "self.maximum_abs_normalized_action_per_hand = [0.0] * 2" in source
    assert "self.maximum_abs_learned_correction_m = [0.0] * 2" in source
    assert "self.maximum_abs_residual_per_joint_rad = [0.0] * 14" in source
    assert "for index in range(2):" in source
    assert "for index in range(14):" in source
    for field in (
        "frame=",
        "left_gap_m=",
        "right_gap_m=",
        "left_contact=",
        "right_contact=",
        "left_force_n=",
        "right_force_n=",
        "root_tilt_deg=",
    ):
        assert field in source


def test_example_screening_payload_is_campaign_compatible() -> None:
    candidate = {
        "path": "/tmp/model_100.pt",
        "sha256": "a" * 64,
        "iteration": 100,
        "episodes": [{"seed": seed, "qualified": False} for seed in (42, 43, 44)],
    }
    payload = {
        "status": "FAIL",
        "diagnostic_valid": True,
        "screening_seeds": [42, 43, 44],
        "candidate_count": 1,
        "screening_episode_count": 3,
        "new_policy_status": "FAIL",
        "best_checkpoint": candidate,
    }
    assert json.loads(json.dumps(payload, allow_nan=False))["screening_episode_count"] == 3
