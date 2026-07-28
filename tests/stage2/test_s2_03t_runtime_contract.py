"""Static/runtime-free contracts for the S2-03T Isaac and RSL-RL integration."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import torch
import yaml

from g1_access_push.stage2.s2_03t_contract import ACTION_JOINT_NAMES, REWARD_SPECS


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / "src/g1_access_push/sim/stage2/s2_03t_actions.py"
MDP = ROOT / "src/g1_access_push/sim/stage2/s2_03t_mdp.py"
ENV_CFG = ROOT / "src/g1_access_push/sim/stage2/s2_03t_env_cfg.py"
ENV = ROOT / "src/g1_access_push/sim/stage2/s2_03t_env.py"
AGENT = ROOT / "src/g1_access_push/sim/stage2/s2_03t_agent_cfg.py"
BOOTSTRAP = ROOT / "src/g1_access_push/sim/stage2/s2_03t_bootstrap.py"
TRAIN = ROOT / "scripts/stage2_isaac/train_s2_03t.py"
EVALUATE = ROOT / "scripts/stage2_isaac/evaluate_s2_03t_checkpoint.py"
EVAL_WRAPPER = ROOT / "scripts/stage2_isaac/run_s2_03t_evaluation_once.sh"
CHECKPOINT = Path(
    "/root/autodl-tmp/robotics/third_party/WBC-AGILE/agile/data/policy/"
    "velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_certified_batched_student_matches_exported_single_environment_for_multiple_steps() -> None:
    policy_single = torch.jit.load(str(CHECKPOINT), map_location="cpu").eval()
    policy_batch = torch.jit.load(str(CHECKPOINT), map_location="cpu").eval()
    policy_single.reset_flat()
    hidden = torch.zeros((1, 1, 256))
    cell = torch.zeros_like(hidden)
    generator = torch.Generator().manual_seed(20260728)
    for _ in range(20):
        policy_input = torch.randn((1, 80), generator=generator)
        expected = policy_single(policy_input[0])
        recurrent, (hidden, cell) = policy_batch.rnn.forward__0(
            policy_batch.normalizer(policy_input.unsqueeze(0)), (hidden, cell)
        )
        actual = policy_batch.actor(recurrent.squeeze(0))
        assert torch.equal(actual[0], expected)


def test_batched_adapter_has_zero_public_dim_and_subset_reset_contract() -> None:
    source = ACTIONS.read_text(encoding="utf-8")
    assert "class FrozenRecurrentLowerBodyAction" in source
    assert "return 0" in source
    assert "self._hidden_state[:, env_ids] = 0.0" in source
    assert "self._cell_state[:, env_ids] = 0.0" in source
    assert "self._previous_policy_actions[env_ids] = 0.0" in source
    assert "policy.rnn.forward__0" in source
    assert "parameter.requires_grad_(False)" in source
    assert sha256(CHECKPOINT) == "a0151975a757a33f0f5ed236d5616e2a98643abe2b36e50ac9ad01b6dd1f6d7e"


def test_arm_action_is_exact_14d_rate_limited_reference_mapping() -> None:
    source = ACTIONS.read_text(encoding="utf-8")
    module = ast.parse(source)
    arm_names = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ARM_JOINT_NAMES" for target in node.targets)
    )
    assert arm_names == ACTION_JOINT_NAMES
    for token in (
        "desired = clipped * float(self.cfg.residual_scale_rad)",
        "desired - self._applied_residual",
        "maximum_residual_change_rad",
        "self._reference + self._applied_residual",
        "joint_limit_margin_rad",
    ):
        assert token in source


def test_manager_based_task_has_exact_dimensions_and_no_training_camera() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    assert "ManagerBasedRLEnvCfg" in source
    assert 'arm_residual = ArmResidualActionCfg(' in source
    assert 'frozen_lower_body = FrozenRecurrentLowerBodyActionCfg(' in source
    assert "CameraCfg" not in source
    assert "episode_length_s: float = 20.0" in source
    assert "self.decimation = 4" in source and "self.sim.dt = 0.005" in source
    assert "self.scene.num_envs" not in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/.*"' in source
    assert "filter_prim_paths_expr=list(BOX_FILTER_EXPRESSIONS)" in source


def test_reward_names_and_weights_match_frozen_yaml() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    contract = yaml.safe_load((ROOT / "configs/stage2/s2_03t_contact_training_contract.yaml").read_text())
    expected = {item["name"]: float(item["weight"]) for item in contract["reward_contract"]["terms"]}
    assert expected == {item.name: item.weight for item in REWARD_SPECS}
    for name, weight in expected.items():
        assert f"{name} = RewTerm(func=mdp.{name}, weight={weight}" in source
    assert "progress_reward" not in source


def test_terminations_are_independent_and_timeout_is_truncation() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    for name in (
        "nonfinite",
        "forbidden_non_palm_box_collision",
        "contact_loss",
        "single_hand_timeout",
        "force_peak",
        "palm_impulse",
        "combined_impulse",
        "force_rate",
        "box_linear_speed",
        "box_angular_speed",
        "box_translation",
        "box_yaw_change",
        "base_excursion",
        "root_height",
        "root_tilt",
        "arm_joint_margin",
        "arm_torque",
    ):
        assert f'{name} = _done("{name}")' in source
    assert "time_out = DoneTerm(func=mdp.time_out, time_out=True)" in source


def test_precontact_reference_captures_full_exact_state_and_restores_on_subset_reset() -> None:
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    environment = ENV.read_text(encoding="utf-8")
    for steps in ("WARMUP_STEPS = 100", "STAND_SETTLE_STEPS = 100", "MOVE_TO_PRECONTACT_STEPS = 150", "PRECONTACT_HOLD_STEPS = 50"):
        assert steps in bootstrap
    for field in (
        "robot_root_state_relative",
        "robot_joint_position",
        "robot_joint_velocity",
        "box_root_state_relative",
        "arm_ik_target",
        "lower_hidden_state",
        "lower_cell_state",
        "previous_lower_policy_action",
    ):
        assert field in environment
        assert field in bootstrap or field in ACTIONS.read_text(encoding="utf-8")
    assert "if reference.ndim == 1:" in ACTIONS.read_text(encoding="utf-8")
    assert "arm.restore_reference(env_ids" in environment
    assert "lower.restore_reference(env_ids" in environment
    assert "runtime_state(self).reset(env_ids)" in environment


def test_agent_and_training_entry_are_clean_no_resume_current_api() -> None:
    agent = AGENT.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")
    for token in (
        "num_steps_per_env = 24",
        "actor_hidden_dims=[256, 128, 64]",
        "critic_hidden_dims=[256, 128, 64]",
        "empirical_normalization = True",
        "clip_actions = 1.0",
        "resume = False",
        "load_run = None",
        "load_checkpoint = None",
    ):
        assert token in agent
    assert "runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)" in train
    assert "runner.load(" not in train
    assert '"model_1999_used": False' in train


def test_actor_evaluation_stops_on_first_done_and_calls_unchanged_evaluator() -> None:
    evaluator = EVALUATE.read_text(encoding="utf-8")
    wrapper = EVAL_WRAPPER.read_text(encoding="utf-8")
    assert "if bool(done[0]):" in evaluator
    assert "terminal_snapshot = env.pop_terminal_snapshot(0)" in evaluator
    assert "break" in evaluator
    assert "post_initial_reset_count\": 0" in evaluator
    assert "scripts/stage2/evaluate_s2_03_attach_only.py" in wrapper
    assert "--enable_cameras" in wrapper


def test_launchers_derive_effective_rc_from_authoritative_json() -> None:
    smoke = (ROOT / "scripts/stage2_isaac/run_s2_03t_smoke_once.sh").read_text(encoding="utf-8")
    train = (ROOT / "scripts/stage2_isaac/run_s2_03t_training_once.sh").read_text(encoding="utf-8")
    evaluate = EVAL_WRAPPER.read_text(encoding="utf-8")
    assert "contract_smoke_result.json\" PASS" in smoke
    assert "training_result.json\" PASS" in train
    assert "runner_status.json\" COMPLETE" in evaluate
    assert "result.json\" PASS FAIL" in evaluate
    for wrapper in (smoke, train, evaluate):
        assert "AUTHORITATIVE_STATUS={status}" in wrapper
        assert "STATUS_RC=${PIPESTATUS[0]}" in wrapper


def test_runtime_metric_values_and_failure_masks_have_disjoint_namespaces() -> None:
    source = MDP.read_text(encoding="utf-8")
    assert "failure_arm_joint_margin" in source
    assert "failure_forbidden_non_palm_box_collision" in source
    assert "COUNTER_TERMINATION_METRIC_NAMES = {\"contact_loss\", \"single_hand_timeout\"}" in source
    assert "name if name in COUNTER_TERMINATION_METRIC_NAMES" in source
    assert "key = metric_name if metric_name in COUNTER_TERMINATION_METRIC_NAMES" in source
    assert "\"termination_reasons\": reasons" in source


def test_original_54_path_hash_snapshot_remains_unchanged() -> None:
    snapshot = Path("/tmp/g1_s2_01_20260727_124751/worktree_before.json")
    assert snapshot.is_file()
    records = json.loads(snapshot.read_text(encoding="utf-8"))
    assert len(records) == 54
    mismatches = []
    for record in records:
        path = ROOT / record["path"]
        if record["file_type"] == "regular" and (not path.is_file() or sha256(path) != record["sha256"]):
            mismatches.append(record["path"])
    assert mismatches == []
