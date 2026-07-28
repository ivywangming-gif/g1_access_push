"""Pure static tests for the redesigned S2-03T contact-policy contract."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from g1_access_push.stage2.s2_03t_redesign_contract import (  # noqa: E402
    ACTION_DIM,
    ACTION_NAME,
    ACTION_ORDER,
    ACTOR_DIM,
    ACTOR_FRAME_COMPONENTS,
    ACTOR_FRAME_DIM,
    ACTOR_HISTORY_FRAMES,
    CONTACTED_HAND_INWARD_CAP_M,
    CORRECTION_RATE_LIMIT_M_PER_CONTROL_STEP,
    CRITIC_DIM,
    CRITIC_PRIVILEGED_APPEND_DIM,
    CRITIC_PRIVILEGED_COMPONENTS,
    CURRICULUM_LEVELS,
    FORCE_HARD_THRESHOLD_N,
    FORCE_MINIMUM_N,
    FORCE_SOFT_MAXIMUM_N,
    NOMINAL_MAXIMUM_DISPLACEMENT_M,
    PROGRESS_CLIP_M,
    VERIFY_STEPS,
    ContractError,
    CurriculumWindow,
    NominalApproachState,
    RewardMode,
    RewardStepInput,
    advance_nominal_approach,
    advance_reward_state,
    build_resolved_config,
    compute_reward_step,
    gap_progress,
    load_yaml,
    map_bilateral_normal_action,
    map_safe_joint_targets,
    observation_contract_payload,
    reset_reward_state,
    reward_landscape_v2,
    safe_force_band_score,
    sha256_file,
    should_promote_curriculum,
    solve_damped_least_squares,
    unsafe_force_excess_squared,
    validate_campaign_execution_manifest,
    validate_contract_bundle,
)

ACTION_PATH = ROOT / "configs/stage2/s2_03t_contact_action_contract.yaml"
REWARD_PATH = ROOT / "configs/stage2/s2_03t_contact_curriculum_reward_contract.yaml"
CURRICULUM_PATH = ROOT / "configs/stage2/s2_03t_contact_curriculum_contract.yaml"
ATTACH_PATH = ROOT / "configs/stage2/s2_03_attach_only.yaml"
LEGACY_TRAINING_PATH = ROOT / "configs/stage2/s2_03t_contact_training_contract.yaml"
GENERATOR = ROOT / "scripts/stage2/generate_s2_03t_redesign_resolved_config.py"


def bundle() -> tuple[dict, dict, dict, dict, dict]:
    return tuple(  # type: ignore[return-value]
        load_yaml(path)
        for path in (ACTION_PATH, REWARD_PATH, CURRICULUM_PATH, ATTACH_PATH, LEGACY_TRAINING_PATH)
    )


def test_yaml_contracts_parse_and_cross_validate() -> None:
    action, reward, curriculum, attach, legacy = bundle()
    gates = validate_contract_bundle(action, reward, curriculum, attach, legacy)
    assert action["action_name"] == ACTION_NAME
    assert action["actor_action"]["dimension"] == ACTION_DIM == 2
    assert tuple(action["actor_action"]["order"]) == ACTION_ORDER
    assert gates["bilateral_contact_threshold_n"] == [1.0, 1.0]
    assert gates["maximum_episode_steps"] == 1000
    assert gates["auto_reset_within_evaluation_episode"] is False


def test_formal_force_motion_and_stillness_gates_are_unchanged() -> None:
    action, reward, curriculum, attach, legacy = bundle()
    gates = validate_contract_bundle(action, reward, curriculum, attach, legacy)
    assert (
        gates["bilateral_verification_steps"]
        == attach["motion"]["bilateral_verification_steps"]
        == 20
    )
    assert gates["attached_hold_steps"] == attach["motion"]["attached_hold_steps"] == 100
    assert gates["per_palm_force_peak_threshold_n"] == FORCE_HARD_THRESHOLD_N == 9.81
    assert gates["per_palm_impulse_threshold_ns"] == 0.981
    assert gates["maximum_box_translation_m"] == 0.005
    assert gates["maximum_box_yaw_change_rad"] == pytest.approx(math.radians(0.5))
    assert gates["maximum_root_tilt_deg"] == 6.2075676918029785
    assert gates["minimum_arm_joint_limit_margin_rad"] == 0.10
    assert gates["maximum_arm_torque_ratio"] == 1.001
    assert gates["pushing_enabled"] is False
    assert gates["planner_enabled"] is False
    assert reward["frozen_scientific_gates"]["thresholds_changed"] is False


def test_gap_progress_is_lazy_signed_deadbanded_clipped_and_mode_gated() -> None:
    progress, previous = gap_progress(None, (0.060, 0.060), RewardMode.APPROACH)
    assert progress == 0.0 and previous == pytest.approx(0.120)
    progress, previous = gap_progress(previous, (0.05975, 0.05975), RewardMode.APPROACH)
    assert progress == pytest.approx(0.0005)
    progress, _ = gap_progress(previous, (0.059753, 0.059753), RewardMode.APPROACH)
    assert progress == 0.0
    progress, _ = gap_progress(previous, (0.061, 0.061), RewardMode.APPROACH)
    assert progress == -PROGRESS_CLIP_M
    progress, updated = gap_progress(previous, (0.01, 0.01), RewardMode.CONTACT_ACQUIRE)
    assert progress == 0.0 and updated == pytest.approx(0.02)


def test_mode_transition_verify_hold_and_onsets_are_once_per_episode() -> None:
    state = reset_reward_state()
    initial = advance_reward_state(state, (0.06, 0.06), left_contact=False, right_contact=False)
    assert initial.reward_mode is initial.state.mode is RewardMode.APPROACH
    left = advance_reward_state(initial.state, (0.0, 0.02), left_contact=True, right_contact=False)
    assert left.reward_mode is left.state.mode is RewardMode.CONTACT_ACQUIRE
    assert left.gap_progress_m == 0.0
    assert left.events.left_contact_onset is True
    repeat_left = advance_reward_state(
        left.state, (0.0, 0.02), left_contact=True, right_contact=False
    )
    assert repeat_left.events.left_contact_onset is False
    both = advance_reward_state(
        repeat_left.state, (0.0, 0.0), left_contact=True, right_contact=True
    )
    assert both.events.right_contact_onset is True
    assert both.events.bilateral_contact_onset is True
    assert both.verify_progress_fraction == pytest.approx(1.0 / VERIFY_STEPS)
    transition = both
    for _ in range(VERIFY_STEPS - 1):
        transition = advance_reward_state(
            transition.state, (0.0, 0.0), left_contact=True, right_contact=True
        )
    assert transition.reward_mode is RewardMode.CONTACT_ACQUIRE
    assert transition.state.mode is RewardMode.VERIFY_HOLD
    hold = advance_reward_state(transition.state, (0.0, 0.0), left_contact=True, right_contact=True)
    assert hold.reward_mode is hold.state.mode is RewardMode.VERIFY_HOLD
    assert hold.state.hold_steps == 1
    assert not any(vars(hold.events).values())


def test_reset_clears_all_reward_history() -> None:
    state = reset_reward_state()
    transition = advance_reward_state(state, (0.0, 0.0), left_contact=True, right_contact=True)
    assert transition.state.previous_gap_sum_m is not None
    assert transition.state.bilateral_onset_awarded is True
    assert reset_reward_state() == type(state)()
    assert reset_reward_state().previous_gap_sum_m is None
    assert reset_reward_state().verify_steps == reset_reward_state().hold_steps == 0


def test_safe_force_band_is_bounded_and_not_monotonic_in_force() -> None:
    assert safe_force_band_score(FORCE_MINIMUM_N - 1.0e-6) == 0.0
    assert safe_force_band_score(FORCE_MINIMUM_N) == 1.0
    assert safe_force_band_score(2.0) == 1.0
    assert safe_force_band_score(FORCE_SOFT_MAXIMUM_N) == 1.0
    assert safe_force_band_score(FORCE_SOFT_MAXIMUM_N + 1.0e-6) == 0.0
    assert safe_force_band_score(FORCE_HARD_THRESHOLD_N) == 0.0
    assert unsafe_force_excess_squared(2.0) == 0.0
    assert 0.0 < unsafe_force_excess_squared(5.0) < 1.0
    assert unsafe_force_excess_squared(FORCE_HARD_THRESHOLD_N) == 1.0


def test_contact_modes_disable_gap_reward_and_hard_push_terms_stay_terminal() -> None:
    acquire = RewardStepInput(
        mode=RewardMode.CONTACT_ACQUIRE,
        surface_gap_m=(0.0, 0.0),
        gap_progress_m=PROGRESS_CLIP_M,
        left_contact=True,
        right_contact=True,
        palm_force_n=(2.0, 2.0),
    )
    result = compute_reward_step(acquire)
    assert result.actual_contribution["gap_progress"] == 0.0
    assert result.actual_contribution["safe_force_band"] > 0.0
    hard = compute_reward_step(replace(acquire, hard_force_terminal=True))
    push = compute_reward_step(replace(acquire, pushing_terminal=True))
    assert hard.actual_contribution["hard_force_terminal"] == -2.0
    assert push.actual_contribution["pushing_terminal"] == -2.0


def test_reward_landscape_satisfies_all_frozen_return_relations() -> None:
    landscape = reward_landscape_v2()
    assert all(landscape["required_relations"].values())
    episodes = landscape["episode_returns"]
    assert episodes["hover_25_mm_1000_steps"] <= 0.0
    assert episodes["bilateral_contact_verify_safe_hold"] > episodes["hover_25_mm_1000_steps"]
    assert (
        episodes["bilateral_contact_verify_safe_hold"]
        > episodes["single_hand_contact_then_timeout"]
    )
    assert episodes["hard_impact_immediate"] < episodes["bilateral_contact_verify_safe_hold"]
    assert episodes["contact_then_pushing"] < episodes["bilateral_contact_verify_safe_hold"]
    assert landscape["states"]["gap_25_mm"]["total"] < 0.0


def test_action_mapping_clips_scales_rate_limits_and_has_correct_normal_sign() -> None:
    mapped = map_bilateral_normal_action((2.0, -2.0), (0.0, 0.0))
    assert mapped.clipped_action == (1.0, -1.0)
    assert mapped.requested_correction_m == pytest.approx((0.005, -0.005))
    assert mapped.applied_correction_m == pytest.approx(
        (CORRECTION_RATE_LIMIT_M_PER_CONTROL_STEP, -CORRECTION_RATE_LIMIT_M_PER_CONTROL_STEP)
    )
    assert mapped.rate_limited == (True, True)
    # Positive object +x is the frozen inward/rear-gap-reducing direction.
    assert mapped.applied_correction_m[0] > 0.0


def test_action_mapping_is_left_right_mirror_equivariant_and_contact_capped() -> None:
    direct = map_bilateral_normal_action((0.25, -0.50), (0.0003, -0.0002))
    mirrored = map_bilateral_normal_action((-0.50, 0.25), (-0.0002, 0.0003))
    assert mirrored.applied_correction_m == pytest.approx(
        tuple(reversed(direct.applied_correction_m))
    )
    contact = map_bilateral_normal_action(
        (1.0, 1.0), (CONTACTED_HAND_INWARD_CAP_M,) * 2, contacted=(True, False)
    )
    assert contact.applied_correction_m[0] == CONTACTED_HAND_INWARD_CAP_M
    assert contact.inward_cap_limited == (True, False)
    assert contact.applied_correction_m[1] > 0.0


def test_nominal_approach_spans_60_mm_and_stops_each_contacted_hand_independently() -> None:
    state = NominalApproachState()
    steps = 0
    while state.displacement_m[0] < 0.060:
        state = advance_nominal_approach(state)
        steps += 1
        assert steps < 1000
    assert state.displacement_m[0] == pytest.approx(state.displacement_m[1])
    assert state.displacement_m[0] <= NOMINAL_MAXIMUM_DISPLACEMENT_M
    frozen_left = state.displacement_m[0]
    next_state = advance_nominal_approach(state, contacted=(True, False))
    assert next_state.displacement_m[0] == frozen_left
    assert next_state.speed_mps[0] == next_state.acceleration_mps2[0] == 0.0
    assert next_state.displacement_m[1] > state.displacement_m[1]
    stopped = advance_nominal_approach(next_state, safety_stop=True)
    assert stopped.displacement_m == next_state.displacement_m
    assert stopped.speed_mps == stopped.acceleration_mps2 == (0.0, 0.0)


def test_damped_least_squares_is_finite_and_rejects_bad_inputs() -> None:
    jacobian = [[0.0] * 14 for _ in range(6)]
    for index in range(6):
        jacobian[index][index] = 1.0
    delta = solve_damped_least_squares(jacobian, [0.001] * 6)
    assert len(delta) == 14 and all(math.isfinite(value) for value in delta)
    assert delta[:6] == pytest.approx([0.001 / (1.0 + 0.01**2)] * 6)
    with pytest.raises(ContractError, match="positive"):
        solve_damped_least_squares(jacobian, [0.001] * 6, damping_lambda=0.0)
    jacobian[0][0] = math.nan
    with pytest.raises(ContractError, match="finite"):
        solve_damped_least_squares(jacobian, [0.001] * 6)


def test_joint_target_mapping_enforces_rate_and_joint_margin() -> None:
    mapped = map_safe_joint_targets([0.0] * 14, [0.5, -0.5] * 7, [-1.0] * 14, [1.0] * 14)
    assert mapped.applied_delta_rad == pytest.approx([0.005, -0.005] * 7)
    assert mapped.minimum_joint_margin_rad == pytest.approx(0.995)
    assert all(mapped.clamped)
    near_limit = map_safe_joint_targets(
        [0.899] + [0.0] * 13,
        [0.005] + [0.0] * 13,
        [-1.0] * 14,
        [1.0] * 14,
    )
    assert near_limit.joint_target_rad[0] == pytest.approx(0.9)
    assert near_limit.minimum_joint_margin_rad == pytest.approx(0.1)
    with pytest.raises(ContractError, match="violates"):
        map_safe_joint_targets([0.91] + [0.0] * 13, [0.0] * 14, [-1.0] * 14, [1.0] * 14)


def test_curriculum_order_and_no_skip_promotion_gate() -> None:
    assert [(item.minimum_gap_m, item.maximum_gap_m) for item in CURRICULUM_LEVELS] == [
        (0.0, 0.005),
        (0.0, 0.010),
        (0.010, 0.030),
        (0.030, 0.060),
    ]
    passing = CurriculumWindow(2048, 0.80, 0.005, 0.001, 0)
    assert should_promote_curriculum(0, passing, pilot=True) is True
    assert should_promote_curriculum(1, passing, pilot=True) is False
    assert should_promote_curriculum(1, passing, pilot=False) is True
    assert (
        should_promote_curriculum(2, replace(passing, success_fraction=0.65), pilot=False) is True
    )
    assert (
        should_promote_curriculum(2, replace(passing, completed_episodes=2047), pilot=False)
        is False
    )
    assert should_promote_curriculum(2, replace(passing, nonfinite_count=1), pilot=False) is False
    assert should_promote_curriculum(3, passing, pilot=False) is False
    curriculum = bundle()[2]
    assert curriculum["sampling"]["full_gap_endpoint_m"] == pytest.approx(0.060)
    assert curriculum["sampling"]["full_gap_endpoint_probability_at_curriculum_3"] == pytest.approx(
        0.05
    )


def test_actor_critic_dimensions_and_privileged_isolation_are_exact() -> None:
    payload = observation_contract_payload()
    assert sum(item.dimension for item in ACTOR_FRAME_COMPONENTS) == ACTOR_FRAME_DIM == 45
    assert ACTOR_FRAME_DIM * ACTOR_HISTORY_FRAMES == ACTOR_DIM == 135
    assert (
        sum(item.dimension for item in CRITIC_PRIVILEGED_COMPONENTS)
        == CRITIC_PRIVILEGED_APPEND_DIM
        == 40
    )
    assert ACTOR_DIM + CRITIC_PRIVILEGED_APPEND_DIM == CRITIC_DIM == 175
    assert payload["actor_dimension"] == 135 and payload["critic_dimension"] == 175
    assert all(item.deployable for item in ACTOR_FRAME_COMPONENTS)
    assert not any(item.deployable for item in CRITIC_PRIVILEGED_COMPONENTS)
    actor_names = {item.name for item in ACTOR_FRAME_COMPONENTS}
    assert "exact_palm_force_n" not in actor_names
    assert "normal_jacobian_authority" not in actor_names
    assert "minimum_joint_margin" not in actor_names


def test_redesign_contract_keeps_prohibitions_and_does_not_import_runtime() -> None:
    action, _, curriculum, _, _ = bundle()
    assert action["frozen_boundaries"]["old_checkpoint_resume_enabled"] is False
    assert action["frozen_boundaries"]["falcon_enabled"] is False
    assert action["frozen_boundaries"]["pushing_enabled"] is False
    assert action["frozen_boundaries"]["planner_enabled"] is False
    assert curriculum["frozen_boundaries"]["formal_s2_03_gates_changed"] is False
    source = (ROOT / "src/g1_access_push/stage2/s2_03t_redesign_contract.py").read_text(
        encoding="utf-8"
    )
    assert "import torch" not in source
    assert "import isaaclab" not in source.lower()


def test_resolved_config_records_sources_landscape_and_frozen_prohibitions() -> None:
    resolved = build_resolved_config(
        action_path=ACTION_PATH,
        reward_path=REWARD_PATH,
        curriculum_path=CURRICULUM_PATH,
        attach_path=ATTACH_PATH,
        legacy_training_path=LEGACY_TRAINING_PATH,
    )
    assert resolved["status"] == "READY_FOR_RUNTIME_INTEGRATION"
    assert resolved["formal_s2_03_gates_unchanged"] is True
    assert all(resolved["reward_landscape_v2"]["required_relations"].values())
    assert resolved["observation"]["actor_dimension"] == 135
    assert resolved["observation"]["critic_dimension"] == 175
    assert not any(resolved["frozen_prohibitions"].values())
    assert len(resolved["source_sha256"]) == 5
    assert all(len(value) == 64 for value in resolved["source_sha256"].values())


def test_shared_campaign_manifest_validator_checks_every_declared_source_sha(
    tmp_path: Path,
) -> None:
    campaign_root = tmp_path / "campaign"
    stage_root = campaign_root / "reachability_smoke"
    reference = campaign_root / "precontact_reference.pt"
    campaign_root.mkdir()
    reference.write_bytes(b"frozen-reference")
    relative_sources = (
        "src/g1_access_push/stage2/s2_03t_redesign_contract.py",
        "configs/stage2/s2_03t_contact_action_contract.yaml",
    )
    execution_sources = {
        relative: {
            "path": str((ROOT / relative).resolve()),
            "sha256": sha256_file(ROOT / relative),
        }
        for relative in relative_sources
    }
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    manifest = {
        "campaign": "S2_03T_REDESIGN_TRAIN_SCREEN_AND_VIDEO_CAMPAIGN",
        "branch": "stage2/s2-03t-reward-action-redesign",
        "run_root": str(campaign_root),
        "implementation_commit": head,
        "reference": {"path": str(reference), "sha256": sha256_file(reference)},
        "frozen_execution": {"action_dim": 2},
        "execution_sources": execution_sources,
    }
    manifest_path = campaign_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = validate_campaign_execution_manifest(
        manifest_path,
        stage_run_root=stage_root,
        reference_path=reference,
        reference_sha256=sha256_file(reference),
        required_source_relatives=relative_sources,
    )
    assert loaded["implementation_commit"] == head

    manifest["execution_sources"][relative_sources[1]]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="CAMPAIGN_EXECUTION_SOURCE_SHA_MISMATCH"):
        validate_campaign_execution_manifest(
            manifest_path,
            stage_run_root=stage_root,
            reference_path=reference,
            reference_sha256=sha256_file(reference),
            required_source_relatives=relative_sources,
        )


def test_generator_atomically_writes_canonical_json_and_check_mode(tmp_path: Path) -> None:
    output = tmp_path / "resolved.json"
    command = [sys.executable, str(GENERATOR), "--output", str(output)]
    generated = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["action"]["actor_action"]["dimension"] == 2
    assert parsed["reward_landscape_v2"]["episode_returns"]["hover_25_mm_1000_steps"] <= 0.0
    checked = subprocess.run(
        [*command, "--check"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert not list(tmp_path.glob("*.tmp"))
    output.write_text("{}\n", encoding="utf-8")
    stale = subprocess.run(
        [*command, "--check"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert stale.returncode == 1


def test_all_yaml_and_generated_numeric_values_are_finite(tmp_path: Path) -> None:
    for path in (ACTION_PATH, REWARD_PATH, CURRICULUM_PATH, ATTACH_PATH, LEGACY_TRAINING_PATH):
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
    output = tmp_path / "resolved.json"
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    text = output.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
