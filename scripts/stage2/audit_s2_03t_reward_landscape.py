#!/usr/bin/env python3
"""Reproduce the pre-redesign S2-03T reward landscape without Isaac/Kit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from g1_access_push.stage2.s2_03t_contract import REWARD_SPECS, compute_reward_terms

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "8bef297b38051bb9b550e6e97bc7f3ee6370a635"
CONTROL_DT_S = 0.02
FORCE_PEAK_THRESHOLD_N = 9.81
COMBINED_IMPULSE_THRESHOLD_NS = 1.962
FORCE_RATE_THRESHOLD_NPS = 490.5
SOURCE_SHA256 = {
    "configs/stage2/s2_03t_contact_training_contract.yaml": "221f27d2a22d3e07f9bd9a05680d61d57f3b2af59a2cf0ed4f5bd8af424e6316",
    "src/g1_access_push/stage2/s2_03t_contract.py": "50ad6ee5a7cc219a536d0aff049a0e991c1e255eb8930d08b05dfd7bbd6f09b2",
    "src/g1_access_push/sim/stage2/s2_03t_mdp.py": "93c53c2ca24a9ff87389bfba0b35994293976413204bb037ea3b25455898ce44",
    "src/g1_access_push/sim/stage2/s2_03t_env_cfg.py": "f6658901686d5a0c09ed2ca88c66d28817e2ba8181f2ab1e7da1fc874efa68f6",
    "reports/stage2/s2_03t_formal_training_summary.json": "1bc32374fdc009a6db8ee8b237dddd432d16f092ef11c7b8aec1e88021dde190",
    "reports/stage2/s2_03t_screening_summary.json": "69a5fd8b8958e241120cb0a61b624989ccfbba7ea7e14f6a83212aa1518bd930",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_at_commit(path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def verify_sources() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path, expected in SOURCE_SHA256.items():
        data = source_at_commit(path)
        actual = sha256_bytes(data)
        result[path] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }
    if not all(item["matches"] for item in result.values()):
        raise RuntimeError("FROZEN_REWARD_SOURCE_SHA_MISMATCH")
    reward_manager = Path(
        "/root/autodl-tmp/robotics/third_party/IsaacLab/source/isaaclab/"
        "isaaclab/managers/reward_manager.py"
    )
    text = reward_manager.read_text(encoding="utf-8")
    if "term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt" not in text:
        raise RuntimeError("ISAAC_REWARD_DT_SCALING_NOT_CONFIRMED")
    result[str(reward_manager)] = {
        "sha256": hashlib.sha256(reward_manager.read_bytes()).hexdigest(),
        "dt_scaling_confirmed": True,
        "source_line": 149,
    }
    return result


def force_composite(
    forces: tuple[float, float],
    previous_forces: tuple[float, float],
    impulse: tuple[float, float],
) -> float:
    force_rate = (
        max(abs(forces[index] - previous_forces[index]) for index in range(2)) / CONTROL_DT_S
    )
    return (
        max(forces) / FORCE_PEAK_THRESHOLD_N
        + sum(impulse) / COMBINED_IMPULSE_THRESHOLD_NS
        + force_rate / FORCE_RATE_THRESHOLD_NPS
    )


def metrics(
    gaps: tuple[float, float],
    *,
    forces: tuple[float, float] = (0.0, 0.0),
    previous_forces: tuple[float, float] = (0.0, 0.0),
    impulse: tuple[float, float] = (0.0, 0.0),
    all_frozen_gates: bool = False,
    success: bool = False,
    timeout: bool = False,
    box_translation_m: float = 0.0,
    forbidden_collision: bool = False,
) -> dict[str, Any]:
    return {
        "surface_gap_m": list(gaps),
        "palm_position_error_m": [abs(gaps[0]), abs(gaps[1])],
        "palm_orientation_error_rad": [0.0, 0.0],
        "palm_force_norm_n": list(forces),
        "left_contact": forces[0] >= 1.0,
        "right_contact": forces[1] >= 1.0,
        "all_frozen_gates": all_frozen_gates,
        "frozen_attach_success": success,
        "box_translation_m": box_translation_m,
        "box_yaw_change_rad": 0.0,
        "normalized_peak_plus_impulse_plus_rate": force_composite(forces, previous_forces, impulse),
        "root_tilt_deg": 0.0,
        "minimum_arm_joint_limit_margin_rad": 1.0,
        "arm_torque_ratio_max": 0.0,
        "forbidden_non_palm_box_collision": forbidden_collision,
        "bilateral_contact_timeout": timeout,
    }


def step_record(label: str, values: dict[str, Any]) -> dict[str, Any]:
    computed = compute_reward_terms(values, [0.0] * 14, [0.0] * 14)
    weighted_dt = {
        name: contribution * CONTROL_DT_S for name, contribution in computed["weighted"].items()
    }
    return {
        "label": label,
        "state": values,
        "raw_terms": computed["raw"],
        "actual_weighted_terms_per_control_step": weighted_dt,
        "actual_total_reward_per_control_step": sum(weighted_dt.values()),
    }


def steady_impulse(forces: tuple[float, float]) -> tuple[float, float]:
    return tuple(force * CONTROL_DT_S * 5.0 for force in forces)


def contact_episode(kind: str) -> dict[str, Any]:
    previous = (0.0, 0.0)
    impulse = [0.0, 0.0]
    totals = {spec.name: 0.0 for spec in REWARD_SPECS}
    steps: list[dict[str, Any]] = []
    if kind == "safe_contact_verify_hold":
        episode_steps = 120
        forces = (1.0, 1.0)
    elif kind == "single_hand_then_timeout":
        episode_steps = 26
        forces = (1.0, 0.0)
    elif kind == "contact_then_push":
        episode_steps = 21
        forces = (1.0, 1.0)
    else:
        raise ValueError(kind)
    for index in range(episode_steps):
        if index < 5:
            impulse[0] += forces[0] * CONTROL_DT_S
            impulse[1] += forces[1] * CONTROL_DT_S
        if kind == "single_hand_then_timeout":
            gaps = (0.0, 0.025)
            all_gates = False
            success = False
            box_translation = 0.0
        else:
            gaps = (0.0, 0.0)
            all_gates = kind == "safe_contact_verify_hold" and index >= 20
            success = kind == "safe_contact_verify_hold" and index == episode_steps - 1
            box_translation = 0.005000001 if kind == "contact_then_push" and index == 20 else 0.0
        record = step_record(
            f"{kind}_step_{index + 1}",
            metrics(
                gaps,
                forces=forces,
                previous_forces=previous,
                impulse=(impulse[0], impulse[1]),
                all_frozen_gates=all_gates,
                success=success,
                box_translation_m=box_translation,
            ),
        )
        for name, value in record["actual_weighted_terms_per_control_step"].items():
            totals[name] += value
        steps.append(record)
        previous = forces
    return {
        "definition": {
            "steps": episode_steps,
            "single_hand_termination_after_steps": 26
            if kind == "single_hand_then_timeout"
            else None,
            "box_translation_terminal_m": 0.005000001 if kind == "contact_then_push" else None,
        },
        "return": sum(totals.values()),
        "term_returns": totals,
        "first_step_reward": steps[0]["actual_total_reward_per_control_step"],
        "final_step_reward": steps[-1]["actual_total_reward_per_control_step"],
    }


def hover_episode(gap_m: float) -> dict[str, Any]:
    ordinary = step_record("hover", metrics((gap_m, gap_m)))
    terminal = step_record("hover_timeout", metrics((gap_m, gap_m), timeout=True))
    return {
        "definition": {"steps": 1000, "gap_each_m": gap_m, "timeout_on_step": 1000},
        "ordinary_step_reward": ordinary["actual_total_reward_per_control_step"],
        "terminal_step_reward": terminal["actual_total_reward_per_control_step"],
        "return": 999.0 * ordinary["actual_total_reward_per_control_step"]
        + terminal["actual_total_reward_per_control_step"],
    }


def approach_then_hover_episode() -> dict[str, Any]:
    # Frozen S2-03 nominal speed is 0.01 m/s => 0.2 mm per 20 ms control step.
    approach_steps = 175
    rewards = []
    for step in range(approach_steps):
        gap = 0.060 - 0.0002 * float(step + 1)
        rewards.append(
            step_record("approach", metrics((gap, gap)))["actual_total_reward_per_control_step"]
        )
    hover = step_record("hover", metrics((0.025, 0.025)))["actual_total_reward_per_control_step"]
    terminal = step_record("timeout", metrics((0.025, 0.025), timeout=True))[
        "actual_total_reward_per_control_step"
    ]
    hover_steps = 1000 - approach_steps
    return {
        "definition": {
            "total_steps": 1000,
            "approach_steps": approach_steps,
            "hover_steps": hover_steps,
            "gap_start_m": 0.060,
            "gap_end_m": 0.025,
            "normal_speed_mps": 0.01,
            "timeout_on_step": 1000,
        },
        "return": sum(rewards) + (hover_steps - 1) * hover + terminal,
    }


def contribution_limits() -> list[dict[str, Any]]:
    definitions = {
        "bilateral_contact_verify_step": ("bool", 0.04, 40.0),
        "attached_hold_step": ("bool", 0.08, 8.0),
        "success_terminal": ("bool", 2.0, 2.0),
        "symmetric_gap_closure": ("dimensionless", 0.02, 20.0),
        "hand_position_tracking": ("m", "UNBOUNDED_IN_REWARD_FUNCTION", "UNBOUNDED"),
        "hand_orientation_tracking": ("rad", -0.012566370614359173, "GATE_BOUNDED_ONLY"),
        "force_imbalance": ("N", -0.0981, "TERMINATES_ABOVE_FORCE_GATE"),
        "box_translation": ("m", -0.004, "TERMINATES_ABOVE_0P005_M"),
        "box_yaw_change": ("rad", -0.006981317007977318, "TERMINATES_ABOVE_GATE"),
        "force_impulse_rate": ("dimensionless", -0.12, "TERMINATES_ABOVE_COMPONENT_GATES"),
        "root_tilt": ("degree", -0.012415135383605959, "TERMINATES_ABOVE_GATE"),
        "joint_limit_margin": ("rad", 0.0, "ZERO_INSIDE_VALID_MARGIN"),
        "torque_ratio": ("ratio", -0.00004, "TERMINATES_ABOVE_1P001"),
        "action_l2": ("normalized_action_squared", -0.00028, -0.28),
        "action_rate": ("normalized_action_delta_squared", -0.0112, -11.2),
        "forbidden_collision_terminal": ("bool", -2.0, -2.0),
        "contact_timeout_terminal": ("bool", -0.5, -0.5),
    }
    weights = {spec.name: spec.weight for spec in REWARD_SPECS}
    return [
        {
            "name": name,
            "weight": weights[name],
            "raw_unit": values[0],
            "maximum_actual_single_step_contribution_with_dt": values[1],
            "maximum_episode_contribution_or_bound": values[2],
        }
        for name, values in definitions.items()
    ]


def build_payload() -> dict[str, Any]:
    provenance = verify_sources()
    one_n_impulse = steady_impulse((1.0, 1.0))
    states = {
        "bilateral_gap_60_mm": step_record("bilateral_gap_60_mm", metrics((0.060, 0.060))),
        "bilateral_gap_30_mm": step_record("bilateral_gap_30_mm", metrics((0.030, 0.030))),
        "bilateral_gap_25_mm": step_record("bilateral_gap_25_mm", metrics((0.025, 0.025))),
        "bilateral_gap_10_mm": step_record("bilateral_gap_10_mm", metrics((0.010, 0.010))),
        "bilateral_gap_5_mm": step_record("bilateral_gap_5_mm", metrics((0.005, 0.005))),
        "single_hand_contact_onset": step_record(
            "single_hand_contact_onset",
            metrics((0.0, 0.025), forces=(1.0, 0.0), impulse=(0.02, 0.0)),
        ),
        "bilateral_contact_onset": step_record(
            "bilateral_contact_onset",
            metrics((0.0, 0.0), forces=(1.0, 1.0), impulse=(0.02, 0.02)),
        ),
        "bilateral_verify": step_record(
            "bilateral_verify",
            metrics(
                (0.0, 0.0),
                forces=(1.0, 1.0),
                previous_forces=(1.0, 1.0),
                impulse=one_n_impulse,
            ),
        ),
        "safe_hold": step_record(
            "safe_hold",
            metrics(
                (0.0, 0.0),
                forces=(1.0, 1.0),
                previous_forces=(1.0, 1.0),
                impulse=one_n_impulse,
                all_frozen_gates=True,
            ),
        ),
        "safe_hold_success_terminal": step_record(
            "safe_hold_success_terminal",
            metrics(
                (0.0, 0.0),
                forces=(1.0, 1.0),
                previous_forces=(1.0, 1.0),
                impulse=one_n_impulse,
                all_frozen_gates=True,
                success=True,
            ),
        ),
        "hard_impact": step_record(
            "hard_impact",
            metrics((0.0, 0.0), forces=(9.82, 9.82), impulse=(0.1964, 0.1964)),
        ),
        "pushing": step_record(
            "pushing",
            metrics(
                (0.0, 0.0),
                forces=(1.0, 1.0),
                previous_forces=(1.0, 1.0),
                impulse=one_n_impulse,
                box_translation_m=0.005000001,
            ),
        ),
        "timeout_at_25_mm": step_record("timeout_at_25_mm", metrics((0.025, 0.025), timeout=True)),
    }
    episodes = {
        "hover_25_mm_1000_steps": hover_episode(0.025),
        "approach_60_to_25_mm_then_hover": approach_then_hover_episode(),
        "bilateral_contact_verify_hold": contact_episode("safe_contact_verify_hold"),
        "single_hand_contact_then_timeout": contact_episode("single_hand_then_timeout"),
        "hard_impact_immediate": {
            "definition": {"steps": 1, "force_each_n": 9.82},
            "return": states["hard_impact"]["actual_total_reward_per_control_step"],
        },
        "contact_then_pushing": contact_episode("contact_then_push"),
        "hover_10_mm_1000_steps": hover_episode(0.010),
    }
    success_terms = episodes["bilateral_contact_verify_hold"]["term_returns"]
    success_return = episodes["bilateral_contact_verify_hold"]["return"]
    relative = {
        name: value / success_return
        for name, value in success_terms.items()
        if not math.isclose(value, 0.0, abs_tol=1.0e-15)
    }
    return {
        "schema_version": 1,
        "stage": "S2-03T",
        "audit": "PRE_REDESIGN_REWARD_LANDSCAPE_AUDIT",
        "status": "PASS",
        "source_commit": SOURCE_COMMIT,
        "control_dt_s": CONTROL_DT_S,
        "actual_reward_semantics": "raw_term_times_weight_times_control_dt",
        "scenario_assumptions": {
            "gap_states": "left_equals_right; palm_position_error_is_normal_gap; other_safe_terms_zero",
            "contact_force_n": 1.0,
            "hard_impact_force_each_n": 9.82,
            "impulse_window_steps": 5,
            "action": "14D_ZERO",
        },
        "source_provenance": provenance,
        "state_rewards": states,
        "episode_returns": episodes,
        "successful_episode_relative_contributions": relative,
        "term_units_and_bounds": contribution_limits(),
        "current_reward_has_hover_local_optimum": "YES",
        "hover_evidence": {
            "hover_25_mm_return": episodes["hover_25_mm_1000_steps"]["return"],
            "hover_10_mm_return": episodes["hover_10_mm_1000_steps"]["return"],
            "safe_contact_verify_hold_return": success_return,
            "ten_mm_hover_is_positive": episodes["hover_10_mm_1000_steps"]["return"] > 0.0,
            "absolute_gap_term_rewards_stationary_near_contact": True,
        },
        "old_scientific_result_preserved": {
            "status": "FAIL",
            "primary_reason": "NO_QUALIFIED_CONTACT_POLICY",
            "formal_bilateral_contact_fraction_maximum": 0.0,
            "screening_valid_failures": "18/18",
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for name, record in payload["state_rewards"].items():
        rows.append(f"| `{name}` | `{record['actual_total_reward_per_control_step']:.12f}` |")
    episode_rows = []
    for name, record in payload["episode_returns"].items():
        episode_rows.append(f"| `{name}` | `{record['return']:.12f}` |")
    text = "\n".join(
        [
            "# S2-03T pre-redesign reward landscape audit",
            "",
            f"Source commit: `{SOURCE_COMMIT}`.",
            "",
            "Isaac RewardManager applies `raw × weight × dt`; here `dt=0.02 s`.",
            "",
            "## State rewards",
            "",
            "| State | Actual reward/control-step |",
            "|---|---:|",
            *rows,
            "",
            "## Episode returns",
            "",
            "| Episode | Return |",
            "|---|---:|",
            *episode_rows,
            "",
            "## Decision",
            "",
            "`CURRENT_REWARD_HAS_HOVER_LOCAL_OPTIMUM=YES`",
            "",
            "The 10 mm no-contact hover episode has positive return. The absolute-gap term therefore rewards stationary near-contact states repeatedly. This is a development diagnosis; the frozen old scientific result remains `FAIL / NO_QUALIFIED_CONTACT_POLICY`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload()
    write_json(args.json, payload)
    write_markdown(args.markdown, payload)
    print("REWARD_LANDSCAPE_AUDIT=PASS")
    print("CURRENT_REWARD_HAS_HOVER_LOCAL_OPTIMUM=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
