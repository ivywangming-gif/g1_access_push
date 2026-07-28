"""Pure static contracts for the S2-03T redesigned PPO training entry."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_CFG = ROOT / "src/g1_access_push/sim/stage2/s2_03t_agent_cfg.py"
TRAIN = ROOT / "scripts/stage2_isaac/train_s2_03t.py"


def _module_assignments(path: Path) -> dict[str, object]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def _runner_class_assignments() -> dict[str, object]:
    module = ast.parse(AGENT_CFG.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "S203TPPORunnerCfg"
    )
    values: dict[str, object] = {}
    for node in runner.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_frozen_pilot_and_formal_invocations_are_exact() -> None:
    assignments = _module_assignments(TRAIN)
    assert assignments["PILOT_CONTRACT"] == {
        "num_envs": 64,
        "max_iterations": 200,
        "save_interval": 25,
        "curriculum_max_level": 1,
    }
    assert assignments["FORMAL_CONTRACT"] == {
        "num_envs": 256,
        "max_iterations": 1200,
        "save_interval": 100,
        "curriculum_max_level": 3,
    }


def test_runner_dimensions_and_clean_initialization_are_frozen() -> None:
    assignments = _module_assignments(TRAIN)
    assert assignments["EXPECTED_ACTOR_INPUT_DIM"] == 135
    assert assignments["EXPECTED_CRITIC_INPUT_DIM"] == 175
    assert assignments["EXPECTED_ACTION_DIM"] == 2
    assert assignments["EXPECTED_ROLLOUT_STEPS_PER_ENV"] == 24
    source = TRAIN.read_text(encoding="utf-8")
    assert "runner.load(" not in source
    assert "agent_cfg.resume = False" in source
    assert "agent_cfg.load_run = None" in source
    assert "agent_cfg.load_checkpoint = None" in source
    assert "agent_cfg.load_optimizer = False" in source
    assert '"checkpoint_loaded": False' in source
    assert '"pilot_checkpoint_used": False' in source
    assert '"model_1999_used": False' in source


def test_campaign_manifest_and_reference_are_validated_before_app_launch() -> None:
    source = TRAIN.read_text(encoding="utf-8")
    validate = source.index("manifest_payload = validate_campaign_manifest(")
    reference = source.index("PRECONTACT_REFERENCE_SHA_MISMATCH")
    launch = source.index("simulation_app = AppLauncher(args).app")
    assert validate < launch
    assert reference < launch
    for token in (
        'parser.add_argument("--campaign-manifest", type=Path, required=True)',
        'parser.add_argument("--curriculum-max-level", type=int, required=True)',
        "CAMPAIGN_MANIFEST_NAME_MISMATCH",
        "CAMPAIGN_FROZEN_EXECUTION_MISMATCH",
        "CAMPAIGN_REFERENCE_SHA_MISMATCH",
        "CAMPAIGN_TRAINING_RUN_ROOT_MISMATCH",
    ):
        assert token in source


def test_curriculum_is_configured_before_reference_install_and_training() -> None:
    source = TRAIN.read_text(encoding="utf-8")
    configure = source.index(
        "env.configure_contact_curriculum(enabled=True, maximum_level=args.curriculum_max_level)"
    )
    install = source.index("env.install_precontact_reference(REFERENCE)")
    learn = source.index(
        "runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)"
    )
    assert configure < install < learn


def test_every_required_training_curve_family_is_persisted() -> None:
    source = TRAIN.read_text(encoding="utf-8")
    required = (
        '"bilateral_contact_fraction"',
        '"bilateral_contact_onset_fraction"',
        '"verify_fraction"',
        '"hold_success_fraction"',
        '"contact_retention_fraction"',
        '"surface_gap_distribution_m"',
        '"force_band_occupancy_fraction"',
        '"hard_force_termination_fraction"',
        '"hard_safety_violation_fraction"',
        '"pushing_fraction"',
        '"fall_fraction"',
        '"maximum_root_tilt_deg"',
        '"maximum_arm_torque_ratio"',
        '"minimum_arm_joint_margin_rad"',
        '"action_saturation_fraction"',
        '"mean_absolute_action_rate"',
        '"episode_return_mean"',
        '"reward_terms_mean"',
        '"iteration_time_s"',
        '"gpu_memory"',
        '"curriculum_promotions"',
        '"nonfinite_fraction"',
    )
    for token in required:
        assert token in source
    assert "drain_training_interval" in source
    assert "class TrainingMetricVecEnvWrapper(RslRlVecEnvWrapper):" in source
    assert "self._reward_term_sum.add_" in source
    assert "wrapper.drain_reward_term_metrics()" in source
    assert "reward_metrics_source" in source
    assert "training_curve.jsonl" in source
    assert "checkpoint_manifest.jsonl" in source


def test_checkpoint_metadata_is_added_by_actual_save_callback() -> None:
    source = TRAIN.read_text(encoding="utf-8")
    assert "class StatusRunner(OnPolicyRunner):" in source
    assert "def save(self, path: str, infos=None):" in source
    assert "super().save(path, infos=infos)" in source
    for token in (
        '"sha256": sha256_file(checkpoint_path)',
        '"iteration_field": int(self.current_learning_iteration)',
        '"actor_input_dim": EXPECTED_ACTOR_INPUT_DIM',
        '"critic_input_dim": EXPECTED_CRITIC_INPUT_DIM',
        '"actor_output_dim": EXPECTED_ACTION_DIM',
        '"serialized_iteration": serialized_iteration',
        '"optimizer_updates_completed": callback_iteration + 1',
        '"saved_after_update": True',
        '"behavior_metric_iteration": (',
    ):
        assert token in source


def test_training_requires_committed_execution_sources_and_complete_finite_runtime() -> None:
    source = TRAIN.read_text(encoding="utf-8")
    for token in (
        "CAMPAIGN_EXECUTION_SOURCES_MISSING",
        "CAMPAIGN_REQUIRED_EXECUTION_SOURCE_MISSING",
        "CAMPAIGN_EXECUTION_SOURCE_SHA_MISMATCH",
        "resolved_training_config.json",
        "runtime_nonfinite_fraction_zero",
        '"seed": args.seed',
        '"clean_actor": True',
    ):
        assert token in source


def test_agent_cfg_has_redesign_formal_defaults_and_no_resume() -> None:
    values = _runner_class_assignments()
    assert values["max_iterations"] == 1200
    assert values["save_interval"] == 100
    assert values["experiment_name"] == "s2_03t_reachable_contact_curriculum"
    assert values["resume"] is False
    assert values["load_run"] is None
    assert values["load_checkpoint"] is None
    assert values["load_optimizer"] is False
