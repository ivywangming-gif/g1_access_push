"""Dependency-light contracts for the redesigned S2-03T contact policy.

The helpers in this module intentionally import neither Isaac Lab nor torch.  They
make the redesigned 2-D action, reward state machine, curriculum, observation
isolation, and frozen S2-03 gates auditable before Kit is started.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = 1
STAGE = "S2-03T"
ACTION_NAME = "HYBRID_NOMINAL_NORMAL_APPROACH_PLUS_LEARNED_BILATERAL_NORMAL_CORRECTION"
ACTION_DIM = 2
ACTION_ORDER = ("left_palm_normal_correction", "right_palm_normal_correction")
ACTION_MIN = -1.0
ACTION_MAX = 1.0
CORRECTION_SCALE_M = 0.005
CORRECTION_RATE_LIMIT_M_PER_CONTROL_STEP = 0.0001
CONTACTED_HAND_INWARD_CAP_M = 0.001
CONTROL_DT_S = 0.02
NOMINAL_SPEED_LIMIT_MPS = 0.01
NOMINAL_ACCELERATION_LIMIT_MPS2 = 0.02
NOMINAL_JERK_LIMIT_MPS3 = 0.10
NOMINAL_MAXIMUM_DISPLACEMENT_M = 0.09
DLS_DAMPING_LAMBDA = 0.01
JOINT_LIMIT_MARGIN_RAD = 0.10
JOINT_TARGET_RATE_LIMIT_RAD_PER_CONTROL_STEP = 0.005
ARM_JOINT_DIM = 14
FORCE_MINIMUM_N = 1.0
FORCE_SOFT_MAXIMUM_N = 3.0
FORCE_HARD_THRESHOLD_N = 9.81
PROGRESS_DEADBAND_M = 0.00001
PROGRESS_CLIP_M = 0.001
VERIFY_STEPS = 20
HOLD_STEPS = 100
MAXIMUM_EPISODE_STEPS = 1000
ACTOR_FRAME_DIM = 45
ACTOR_HISTORY_FRAMES = 3
ACTOR_DIM = 135
CRITIC_PRIVILEGED_APPEND_DIM = 40
CRITIC_DIM = 175
EXPECTED_CAMPAIGN_NAME = "S2_03T_REDESIGN_TRAIN_SCREEN_AND_VIDEO_CAMPAIGN"
EXPECTED_CAMPAIGN_BRANCH = "stage2/s2-03t-reward-action-redesign"


class ContractError(ValueError):
    """Raised when a value violates the redesigned S2-03T contract."""


class RewardMode(StrEnum):
    APPROACH = "APPROACH"
    CONTACT_ACQUIRE = "CONTACT_ACQUIRE"
    VERIFY_HOLD = "VERIFY_HOLD"


@dataclass(frozen=True)
class ObservationComponent:
    name: str
    dimension: int
    deployable: bool


ACTOR_FRAME_COMPONENTS = (
    ObservationComponent("root_angular_velocity_body", 3, True),
    ObservationComponent("projected_gravity_body", 3, True),
    ObservationComponent("arm_joint_position_error_to_reference", 14, True),
    ObservationComponent("arm_joint_velocity", 14, True),
    ObservationComponent("palm_surface_gap", 2, True),
    ObservationComponent("deployable_contact_latch", 2, True),
    ObservationComponent("nominal_displacement", 2, True),
    ObservationComponent("reward_mode_one_hot", 3, True),
    ObservationComponent("previous_action", 2, True),
)

CRITIC_PRIVILEGED_COMPONENTS = (
    ObservationComponent("exact_palm_force_n", 2, False),
    ObservationComponent("contact_force_rate_nps", 1, False),
    ObservationComponent("palm_impulse_ns", 2, False),
    ObservationComponent("palm_position_error", 6, False),
    ObservationComponent("palm_orientation_error", 6, False),
    ObservationComponent("box_translation_vector", 3, False),
    ObservationComponent("box_yaw_change", 1, False),
    ObservationComponent("box_linear_velocity", 3, False),
    ObservationComponent("box_angular_velocity", 3, False),
    ObservationComponent("base_xy_excursion", 1, False),
    ObservationComponent("root_height", 1, False),
    ObservationComponent("root_tilt", 1, False),
    ObservationComponent("minimum_joint_margin", 1, False),
    ObservationComponent("maximum_torque_ratio", 1, False),
    ObservationComponent("normal_jacobian_authority", 2, False),
    ObservationComponent("joint_target_clamp_flags", 2, False),
    ObservationComponent("curriculum_one_hot", 4, False),
)


@dataclass(frozen=True)
class RewardTermSpec:
    name: str
    weight: float
    raw_unit: str
    enabled_modes: tuple[RewardMode, ...]
    maximum_actual_step_contribution: float | str
    event_once_per_episode: bool = False


ALL_MODES = tuple(RewardMode)
CONTACT_MODES = (RewardMode.CONTACT_ACQUIRE, RewardMode.VERIFY_HOLD)

REWARD_TERM_SPECS = (
    RewardTermSpec("gap_progress", 250.0, "m", (RewardMode.APPROACH,), 0.005),
    RewardTermSpec("approach_symmetry", 50.0, "m", (RewardMode.APPROACH,), 0.001),
    RewardTermSpec("left_contact_onset", 1.0, "event", (RewardMode.CONTACT_ACQUIRE,), 0.02, True),
    RewardTermSpec("right_contact_onset", 1.0, "event", (RewardMode.CONTACT_ACQUIRE,), 0.02, True),
    RewardTermSpec(
        "bilateral_contact_onset", 3.0, "event", (RewardMode.CONTACT_ACQUIRE,), 0.06, True
    ),
    RewardTermSpec(
        "verify_progress", 4.0, "verified_fraction", (RewardMode.CONTACT_ACQUIRE,), 0.004
    ),
    RewardTermSpec("safe_force_band", 1.0, "dimensionless", (RewardMode.CONTACT_ACQUIRE,), 0.02),
    RewardTermSpec("single_contact_step", -0.5, "bool", (RewardMode.CONTACT_ACQUIRE,), -0.01),
    RewardTermSpec("contact_retention", 1.0, "bool", (RewardMode.VERIFY_HOLD,), 0.02),
    RewardTermSpec("attached_hold_step", 4.0, "bool", (RewardMode.VERIFY_HOLD,), 0.08),
    RewardTermSpec("force_balance", 1.0, "dimensionless", (RewardMode.VERIFY_HOLD,), 0.02),
    RewardTermSpec("success_terminal", 100.0, "event", (RewardMode.VERIFY_HOLD,), 2.0, True),
    RewardTermSpec("time_cost", -0.01, "control_step", ALL_MODES, -0.0002),
    RewardTermSpec("hand_orientation_tracking", -0.05, "rad", ALL_MODES, "GATE_BOUNDED"),
    RewardTermSpec("unsafe_force", -4.0, "squared_normalized_excess", CONTACT_MODES, -0.16),
    RewardTermSpec("force_impulse_rate", -2.0, "dimensionless", CONTACT_MODES, "GATE_BOUNDED"),
    RewardTermSpec("force_imbalance", -0.5, "N", (RewardMode.VERIFY_HOLD,), "GATE_BOUNDED"),
    RewardTermSpec("box_translation", -40.0, "m", ALL_MODES, "GATE_BOUNDED"),
    RewardTermSpec("box_yaw_change", -40.0, "rad", ALL_MODES, "GATE_BOUNDED"),
    RewardTermSpec("root_risk", -1.0, "degree_above_5", ALL_MODES, "GATE_BOUNDED"),
    RewardTermSpec("joint_limit_margin", -5.0, "rad", ALL_MODES, "GATE_BOUNDED"),
    RewardTermSpec("torque_ratio", -2.0, "ratio", ALL_MODES, "GATE_BOUNDED"),
    RewardTermSpec("action_l2", -0.002, "normalized_action_squared", ALL_MODES, -0.00008),
    RewardTermSpec("action_rate", -0.01, "normalized_action_delta_squared", ALL_MODES, -0.0016),
    RewardTermSpec("hard_force_terminal", -100.0, "event", ALL_MODES, -2.0, True),
    RewardTermSpec("pushing_terminal", -100.0, "event", ALL_MODES, -2.0, True),
    RewardTermSpec("contact_loss_terminal", -25.0, "event", (RewardMode.VERIFY_HOLD,), -0.5, True),
    RewardTermSpec("forbidden_collision_terminal", -100.0, "event", ALL_MODES, -2.0, True),
    RewardTermSpec("contact_timeout_terminal", -25.0, "event", ALL_MODES, -0.5, True),
)


@dataclass(frozen=True)
class RewardState:
    mode: RewardMode = RewardMode.APPROACH
    previous_gap_sum_m: float | None = None
    left_onset_awarded: bool = False
    right_onset_awarded: bool = False
    bilateral_onset_awarded: bool = False
    verify_steps: int = 0
    hold_steps: int = 0


@dataclass(frozen=True)
class RewardEvents:
    left_contact_onset: bool = False
    right_contact_onset: bool = False
    bilateral_contact_onset: bool = False


@dataclass(frozen=True)
class RewardTransition:
    state: RewardState
    reward_mode: RewardMode
    gap_progress_m: float
    events: RewardEvents
    verify_progress_fraction: float


@dataclass(frozen=True)
class RewardStepInput:
    mode: RewardMode
    surface_gap_m: tuple[float, float]
    gap_progress_m: float = 0.0
    left_contact: bool = False
    right_contact: bool = False
    left_contact_onset: bool = False
    right_contact_onset: bool = False
    bilateral_contact_onset: bool = False
    verify_progress_fraction: float = 0.0
    all_frozen_gates: bool = True
    frozen_attach_success: bool = False
    palm_force_n: tuple[float, float] = (0.0, 0.0)
    hand_orientation_error_rad: tuple[float, float] = (0.0, 0.0)
    normalized_force_impulse_rate: float = 0.0
    box_translation_m: float = 0.0
    box_yaw_change_rad: float = 0.0
    root_tilt_deg: float = 0.0
    minimum_arm_joint_limit_margin_rad: float = 0.5
    maximum_arm_torque_ratio: float = 0.0
    action: tuple[float, float] = (0.0, 0.0)
    previous_action: tuple[float, float] = (0.0, 0.0)
    hard_force_terminal: bool = False
    pushing_terminal: bool = False
    contact_loss_terminal: bool = False
    forbidden_collision_terminal: bool = False
    contact_timeout_terminal: bool = False


@dataclass(frozen=True)
class RewardBreakdown:
    raw: dict[str, float]
    actual_contribution: dict[str, float]
    total: float


@dataclass(frozen=True)
class ActionMapping:
    clipped_action: tuple[float, float]
    requested_correction_m: tuple[float, float]
    applied_correction_m: tuple[float, float]
    rate_limited: tuple[bool, bool]
    inward_cap_limited: tuple[bool, bool]


@dataclass(frozen=True)
class NominalApproachState:
    displacement_m: tuple[float, float] = (0.0, 0.0)
    speed_mps: tuple[float, float] = (0.0, 0.0)
    acceleration_mps2: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class JointTargetMapping:
    joint_target_rad: tuple[float, ...]
    requested_delta_rad: tuple[float, ...]
    applied_delta_rad: tuple[float, ...]
    clamped: tuple[bool, ...]
    minimum_joint_margin_rad: float


@dataclass(frozen=True)
class CurriculumLevel:
    level_id: int
    name: str
    minimum_gap_m: float
    maximum_gap_m: float
    success_threshold: float | None


CURRICULUM_LEVELS = (
    CurriculumLevel(0, "CURRICULUM_0", 0.0, 0.005, 0.80),
    CurriculumLevel(1, "CURRICULUM_1", 0.0, 0.010, 0.75),
    CurriculumLevel(2, "CURRICULUM_2", 0.010, 0.030, 0.65),
    CurriculumLevel(3, "CURRICULUM_3", 0.030, 0.060, None),
)


@dataclass(frozen=True)
class CurriculumWindow:
    completed_episodes: int
    success_fraction: float
    hard_safety_violation_fraction: float
    pushing_violation_fraction: float
    nonfinite_count: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of an immutable source artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_campaign_execution_manifest(
    manifest_path: Path,
    *,
    stage_run_root: Path,
    reference_path: Path,
    reference_sha256: str,
    required_source_relatives: Sequence[str],
) -> dict[str, Any]:
    """Validate campaign identity, commit, reference, and every declared source SHA."""

    try:
        payload = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ContractError("CAMPAIGN_MANIFEST_UNREADABLE") from exc
    if not isinstance(payload, dict) or payload.get("campaign") != EXPECTED_CAMPAIGN_NAME:
        raise ContractError("CAMPAIGN_MANIFEST_IDENTITY_MISMATCH")
    if payload.get("branch") != EXPECTED_CAMPAIGN_BRANCH:
        raise ContractError("CAMPAIGN_MANIFEST_BRANCH_MISMATCH")
    campaign_root = Path(str(payload.get("run_root", ""))).resolve()
    resolved_stage_root = stage_run_root.resolve()
    if campaign_root == resolved_stage_root or campaign_root not in resolved_stage_root.parents:
        raise ContractError("CAMPAIGN_STAGE_RUN_ROOT_MISMATCH")
    reference = payload.get("reference")
    resolved_reference = reference_path.resolve()
    if (
        not isinstance(reference, Mapping)
        or Path(str(reference.get("path", ""))).resolve() != resolved_reference
        or reference.get("sha256") != reference_sha256
        or not resolved_reference.is_file()
        or sha256_file(resolved_reference) != reference_sha256
    ):
        raise ContractError("CAMPAIGN_REFERENCE_MISMATCH")
    frozen = payload.get("frozen_execution")
    if not isinstance(frozen, Mapping) or frozen.get("action_dim") != ACTION_DIM:
        raise ContractError("CAMPAIGN_FROZEN_ACTION_MISMATCH")
    implementation_commit = payload.get("implementation_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise ContractError("CAMPAIGN_IMPLEMENTATION_COMMIT_INVALID")
    repository = Path(__file__).resolve().parents[3]
    try:
        current_head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
            timeout=30.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError("CAMPAIGN_IMPLEMENTATION_HEAD_UNREADABLE") from exc
    if current_head != implementation_commit:
        raise ContractError("CAMPAIGN_IMPLEMENTATION_HEAD_MISMATCH")
    sources = payload.get("execution_sources")
    if not isinstance(sources, Mapping) or not set(required_source_relatives).issubset(sources):
        raise ContractError("CAMPAIGN_REQUIRED_EXECUTION_SOURCE_MISSING")
    for relative, record in sources.items():
        relative_path = Path(str(relative))
        if (
            not isinstance(relative, str)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not isinstance(record, Mapping)
        ):
            raise ContractError("CAMPAIGN_EXECUTION_SOURCE_RECORD_INVALID")
        source = (repository / relative_path).resolve()
        if (
            Path(str(record.get("path", ""))).resolve() != source
            or not source.is_file()
            or sha256_file(source) != record.get("sha256")
        ):
            raise ContractError(f"CAMPAIGN_EXECUTION_SOURCE_SHA_MISMATCH:{relative}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject non-mapping roots."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{path} must contain a YAML mapping")
    return payload


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ContractError(f"{name} must have exactly two elements")
    return _finite(values[0], f"{name}[0]"), _finite(values[1], f"{name}[1]")


def reset_reward_state() -> RewardState:
    """Return the only valid post-reset reward history."""

    return RewardState()


def gap_progress(
    previous_gap_sum_m: float | None,
    current_surface_gap_m: Sequence[float],
    mode: RewardMode,
) -> tuple[float, float]:
    """Compute clipped progress and the next history value.

    The first post-reset sample is lazy-initialized and yields zero.  History is
    updated in every mode, while the returned reward signal is zero outside
    ``APPROACH``.
    """

    current = sum(_pair(current_surface_gap_m, "current_surface_gap_m"))
    if previous_gap_sum_m is None:
        return 0.0, current
    previous = _finite(previous_gap_sum_m, "previous_gap_sum_m")
    if mode is not RewardMode.APPROACH:
        return 0.0, current
    progress = previous - current
    if abs(progress) <= PROGRESS_DEADBAND_M:
        progress = 0.0
    progress = max(-PROGRESS_CLIP_M, min(PROGRESS_CLIP_M, progress))
    return progress, current


def advance_reward_state(
    state: RewardState,
    surface_gap_m: Sequence[float],
    *,
    left_contact: bool,
    right_contact: bool,
    all_frozen_gates: bool = True,
) -> RewardTransition:
    """Advance mode/history exactly once for one control step."""

    if state.verify_steps < 0 or state.hold_steps < 0:
        raise ContractError("reward counters must be non-negative")
    both = bool(left_contact and right_contact)
    any_contact = bool(left_contact or right_contact)
    reward_mode = state.mode
    if reward_mode is RewardMode.APPROACH and any_contact:
        reward_mode = RewardMode.CONTACT_ACQUIRE

    progress, current_sum = gap_progress(state.previous_gap_sum_m, surface_gap_m, reward_mode)
    left_onset = bool(left_contact and not state.left_onset_awarded)
    right_onset = bool(right_contact and not state.right_onset_awarded)
    bilateral_onset = bool(both and not state.bilateral_onset_awarded)

    verify_steps = state.verify_steps
    hold_steps = state.hold_steps
    next_mode = reward_mode
    verify_fraction = 0.0
    if reward_mode is RewardMode.CONTACT_ACQUIRE:
        if both and all_frozen_gates:
            verify_steps += 1
            verify_fraction = 1.0 / VERIFY_STEPS
        else:
            verify_steps = 0
        if verify_steps >= VERIFY_STEPS:
            next_mode = RewardMode.VERIFY_HOLD
    elif reward_mode is RewardMode.VERIFY_HOLD:
        if both and all_frozen_gates:
            hold_steps += 1

    next_state = RewardState(
        mode=next_mode,
        previous_gap_sum_m=current_sum,
        left_onset_awarded=state.left_onset_awarded or bool(left_contact),
        right_onset_awarded=state.right_onset_awarded or bool(right_contact),
        bilateral_onset_awarded=state.bilateral_onset_awarded or both,
        verify_steps=verify_steps,
        hold_steps=hold_steps,
    )
    return RewardTransition(
        state=next_state,
        reward_mode=reward_mode,
        gap_progress_m=progress,
        events=RewardEvents(left_onset, right_onset, bilateral_onset),
        verify_progress_fraction=verify_fraction,
    )


def safe_force_band_score(force_n: float) -> float:
    """Reward only the finite closed interval [1, 3] N."""

    force = _finite(force_n, "force_n")
    return float(FORCE_MINIMUM_N <= force <= FORCE_SOFT_MAXIMUM_N)


def unsafe_force_excess_squared(force_n: float) -> float:
    """Return a bounded quadratic excess above the 3 N soft limit."""

    force = _finite(force_n, "force_n")
    span = FORCE_HARD_THRESHOLD_N - FORCE_SOFT_MAXIMUM_N
    normalized = max(0.0, force - FORCE_SOFT_MAXIMUM_N) / span
    return min(1.0, normalized) ** 2


def _mode_one_hot(mode: RewardMode) -> tuple[float, float, float]:
    return tuple(float(mode is candidate) for candidate in RewardMode)  # type: ignore[return-value]


def compute_reward_step(step: RewardStepInput) -> RewardBreakdown:
    """Compute actual RewardManager contributions, including ``control_dt``."""

    mode = RewardMode(step.mode)
    gaps = _pair(step.surface_gap_m, "surface_gap_m")
    forces = _pair(step.palm_force_n, "palm_force_n")
    orientation = _pair(step.hand_orientation_error_rad, "hand_orientation_error_rad")
    action = _pair(step.action, "action")
    previous_action = _pair(step.previous_action, "previous_action")
    both = bool(step.left_contact and step.right_contact)
    single = bool(step.left_contact) != bool(step.right_contact)
    safe_score = 0.5 * sum(safe_force_band_score(force) for force in forces)
    unsafe_force = sum(unsafe_force_excess_squared(force) for force in forces)
    max_force = max(forces[0], forces[1], FORCE_MINIMUM_N)
    force_balance = max(0.0, 1.0 - abs(forces[0] - forces[1]) / max_force)
    raw = {
        "gap_progress": _finite(step.gap_progress_m, "gap_progress_m"),
        # Symmetry is a zero-at-perfect, non-positive penalty.  Therefore a
        # symmetric hover cannot earn a persistent positive shaping reward.
        "approach_symmetry": -min(PROGRESS_CLIP_M, abs(gaps[0] - gaps[1])),
        "left_contact_onset": float(step.left_contact_onset),
        "right_contact_onset": float(step.right_contact_onset),
        "bilateral_contact_onset": float(step.bilateral_contact_onset),
        "verify_progress": max(
            0.0, min(1.0, _finite(step.verify_progress_fraction, "verify_progress"))
        ),
        "safe_force_band": safe_score,
        "single_contact_step": float(single),
        "contact_retention": float(both),
        "attached_hold_step": float(both and step.all_frozen_gates),
        "force_balance": force_balance,
        "success_terminal": float(step.frozen_attach_success),
        "time_cost": 1.0,
        "hand_orientation_tracking": abs(orientation[0]) + abs(orientation[1]),
        "unsafe_force": unsafe_force,
        "force_impulse_rate": max(
            0.0, _finite(step.normalized_force_impulse_rate, "normalized_force_impulse_rate")
        ),
        "force_imbalance": abs(forces[0] - forces[1]),
        "box_translation": abs(_finite(step.box_translation_m, "box_translation_m")),
        "box_yaw_change": abs(_finite(step.box_yaw_change_rad, "box_yaw_change_rad")),
        "root_risk": max(0.0, _finite(step.root_tilt_deg, "root_tilt_deg") - 5.0),
        "joint_limit_margin": max(
            0.0,
            JOINT_LIMIT_MARGIN_RAD
            - _finite(
                step.minimum_arm_joint_limit_margin_rad, "minimum_arm_joint_limit_margin_rad"
            ),
        ),
        "torque_ratio": max(
            0.0, _finite(step.maximum_arm_torque_ratio, "maximum_arm_torque_ratio") - 1.0
        ),
        "action_l2": sum(value * value for value in action),
        "action_rate": sum(
            (value - previous) ** 2 for value, previous in zip(action, previous_action, strict=True)
        ),
        "hard_force_terminal": float(step.hard_force_terminal),
        "pushing_terminal": float(step.pushing_terminal),
        "contact_loss_terminal": float(step.contact_loss_terminal),
        "forbidden_collision_terminal": float(step.forbidden_collision_terminal),
        "contact_timeout_terminal": float(step.contact_timeout_terminal),
    }
    contribution: dict[str, float] = {}
    for spec in REWARD_TERM_SPECS:
        active = mode in spec.enabled_modes
        contribution[spec.name] = raw[spec.name] * spec.weight * CONTROL_DT_S if active else 0.0
    return RewardBreakdown(
        raw=raw, actual_contribution=contribution, total=sum(contribution.values())
    )


def map_bilateral_normal_action(
    action: Sequence[float],
    previous_correction_m: Sequence[float],
    *,
    contacted: Sequence[bool] = (False, False),
) -> ActionMapping:
    """Map normalized left/right commands to safe rate-limited normal corrections.

    Positive correction is object +x and reduces the certified rear-surface gap.
    Swapping left/right inputs swaps outputs exactly; there is no cross-hand mixing.
    """

    action_v = _pair(action, "action")
    previous_v = _pair(previous_correction_m, "previous_correction_m")
    if len(contacted) != 2:
        raise ContractError("contacted must have exactly two elements")
    clipped: list[float] = []
    requested: list[float] = []
    applied: list[float] = []
    rate_limited: list[bool] = []
    cap_limited: list[bool] = []
    for index in range(ACTION_DIM):
        normalized = min(ACTION_MAX, max(ACTION_MIN, action_v[index]))
        request = normalized * CORRECTION_SCALE_M
        capped = min(CONTACTED_HAND_INWARD_CAP_M, request) if bool(contacted[index]) else request
        delta = capped - previous_v[index]
        bounded_delta = max(
            -CORRECTION_RATE_LIMIT_M_PER_CONTROL_STEP,
            min(CORRECTION_RATE_LIMIT_M_PER_CONTROL_STEP, delta),
        )
        result = previous_v[index] + bounded_delta
        clipped.append(normalized)
        requested.append(request)
        applied.append(result)
        rate_limited.append(not math.isclose(delta, bounded_delta, rel_tol=0.0, abs_tol=1.0e-15))
        cap_limited.append(not math.isclose(request, capped, rel_tol=0.0, abs_tol=1.0e-15))
    return ActionMapping(
        clipped_action=(clipped[0], clipped[1]),
        requested_correction_m=(requested[0], requested[1]),
        applied_correction_m=(applied[0], applied[1]),
        rate_limited=(rate_limited[0], rate_limited[1]),
        inward_cap_limited=(cap_limited[0], cap_limited[1]),
    )


def advance_nominal_approach(
    state: NominalApproachState,
    *,
    contacted: Sequence[bool] = (False, False),
    safety_stop: bool = False,
) -> NominalApproachState:
    """Advance the independent jerk-limited nominal approach once."""

    displacement = _pair(state.displacement_m, "displacement_m")
    speed = _pair(state.speed_mps, "speed_mps")
    acceleration = _pair(state.acceleration_mps2, "acceleration_mps2")
    if len(contacted) != 2:
        raise ContractError("contacted must have exactly two elements")
    next_displacement: list[float] = []
    next_speed: list[float] = []
    next_acceleration: list[float] = []
    stop_all = safety_stop or (bool(contacted[0]) and bool(contacted[1]))
    for index in range(ACTION_DIM):
        if (
            stop_all
            or bool(contacted[index])
            or displacement[index] >= NOMINAL_MAXIMUM_DISPLACEMENT_M
        ):
            next_displacement.append(min(NOMINAL_MAXIMUM_DISPLACEMENT_M, displacement[index]))
            next_speed.append(0.0)
            next_acceleration.append(0.0)
            continue
        target_acceleration = (
            NOMINAL_ACCELERATION_LIMIT_MPS2 if speed[index] < NOMINAL_SPEED_LIMIT_MPS else 0.0
        )
        jerk_delta = NOMINAL_JERK_LIMIT_MPS3 * CONTROL_DT_S
        acceleration_i = acceleration[index] + max(
            -jerk_delta, min(jerk_delta, target_acceleration - acceleration[index])
        )
        acceleration_i = max(0.0, min(NOMINAL_ACCELERATION_LIMIT_MPS2, acceleration_i))
        speed_i = min(NOMINAL_SPEED_LIMIT_MPS, speed[index] + acceleration_i * CONTROL_DT_S)
        displacement_i = min(
            NOMINAL_MAXIMUM_DISPLACEMENT_M, displacement[index] + speed_i * CONTROL_DT_S
        )
        next_displacement.append(displacement_i)
        next_speed.append(speed_i)
        next_acceleration.append(acceleration_i)
    return NominalApproachState(
        displacement_m=(next_displacement[0], next_displacement[1]),
        speed_mps=(next_speed[0], next_speed[1]),
        acceleration_mps2=(next_acceleration[0], next_acceleration[1]),
    )


def solve_damped_least_squares(
    jacobian: Sequence[Sequence[float]],
    task_error: Sequence[float],
    *,
    damping_lambda: float = DLS_DAMPING_LAMBDA,
) -> tuple[float, ...]:
    """Solve ``J.T @ inv(J @ J.T + lambda^2 I) @ error`` with finite checks."""

    damping = _finite(damping_lambda, "damping_lambda")
    if damping <= 0.0:
        raise ContractError("damping_lambda must be positive")
    matrix = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(task_error, dtype=np.float64)
    if matrix.ndim != 2 or error.ndim != 1 or matrix.shape[0] != error.shape[0]:
        raise ContractError("jacobian rows must match the 1-D task error")
    if matrix.shape[1] != ARM_JOINT_DIM:
        raise ContractError(f"jacobian must have {ARM_JOINT_DIM} joint columns")
    if not np.isfinite(matrix).all() or not np.isfinite(error).all():
        raise ContractError("jacobian and task error must be finite")
    regularized = matrix @ matrix.T + (damping * damping) * np.eye(matrix.shape[0])
    try:
        delta = matrix.T @ np.linalg.solve(regularized, error)
    except np.linalg.LinAlgError as exc:
        raise ContractError("damped least-squares solve failed") from exc
    if delta.shape != (ARM_JOINT_DIM,) or not np.isfinite(delta).all():
        raise ContractError("damped least-squares output must be finite 14-D")
    return tuple(float(value) for value in delta)


def map_safe_joint_targets(
    current_joint_position_rad: Sequence[float],
    requested_delta_rad: Sequence[float],
    hard_lower_limit_rad: Sequence[float],
    hard_upper_limit_rad: Sequence[float],
) -> JointTargetMapping:
    """Apply the frozen target-rate and joint-margin clips to a 14-D DLS step."""

    vectors = []
    for values, name in (
        (current_joint_position_rad, "current_joint_position_rad"),
        (requested_delta_rad, "requested_delta_rad"),
        (hard_lower_limit_rad, "hard_lower_limit_rad"),
        (hard_upper_limit_rad, "hard_upper_limit_rad"),
    ):
        if len(values) != ARM_JOINT_DIM:
            raise ContractError(f"{name} must have {ARM_JOINT_DIM} elements")
        vectors.append(
            tuple(_finite(value, f"{name}[{index}]") for index, value in enumerate(values))
        )
    current, requested, lower, upper = vectors
    targets: list[float] = []
    applied: list[float] = []
    clamped: list[bool] = []
    margins: list[float] = []
    for index in range(ARM_JOINT_DIM):
        safe_lower = lower[index] + JOINT_LIMIT_MARGIN_RAD
        safe_upper = upper[index] - JOINT_LIMIT_MARGIN_RAD
        if safe_lower > safe_upper or not safe_lower <= current[index] <= safe_upper:
            raise ContractError(
                f"joint {index} current position violates the {JOINT_LIMIT_MARGIN_RAD} rad margin"
            )
        rate_delta = max(
            -JOINT_TARGET_RATE_LIMIT_RAD_PER_CONTROL_STEP,
            min(JOINT_TARGET_RATE_LIMIT_RAD_PER_CONTROL_STEP, requested[index]),
        )
        raw_target = current[index] + rate_delta
        target = min(safe_upper, max(safe_lower, raw_target))
        targets.append(target)
        applied.append(target - current[index])
        clamped.append(
            not math.isclose(
                target, current[index] + requested[index], rel_tol=0.0, abs_tol=1.0e-15
            )
        )
        margins.append(min(target - lower[index], upper[index] - target))
    return JointTargetMapping(
        joint_target_rad=tuple(targets),
        requested_delta_rad=tuple(requested),
        applied_delta_rad=tuple(applied),
        clamped=tuple(clamped),
        minimum_joint_margin_rad=min(margins),
    )


def should_promote_curriculum(
    source_level: int,
    window: CurriculumWindow,
    *,
    pilot: bool,
) -> bool:
    """Apply the frozen no-skip promotion gate to one completed-episode window."""

    if source_level not in range(len(CURRICULUM_LEVELS)):
        raise ContractError(f"unknown curriculum level {source_level}")
    cap = 1 if pilot else 3
    if source_level >= cap:
        return False
    level = CURRICULUM_LEVELS[source_level]
    if level.success_threshold is None:
        return False
    values = (
        window.success_fraction,
        window.hard_safety_violation_fraction,
        window.pushing_violation_fraction,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ContractError("curriculum fractions must be finite and in [0, 1]")
    return bool(
        window.completed_episodes >= 2048
        and window.success_fraction >= level.success_threshold
        and window.hard_safety_violation_fraction <= 0.005
        and window.pushing_violation_fraction <= 0.001
        and window.nonfinite_count == 0
    )


def observation_contract_payload() -> dict[str, Any]:
    """Return actor/critic dimensions and the privileged-information boundary."""

    actor_frame = sum(component.dimension for component in ACTOR_FRAME_COMPONENTS)
    privileged = sum(component.dimension for component in CRITIC_PRIVILEGED_COMPONENTS)
    if actor_frame != ACTOR_FRAME_DIM or privileged != CRITIC_PRIVILEGED_APPEND_DIM:
        raise AssertionError("internal observation dimensions are inconsistent")
    return {
        "actor_frame_components": [component.__dict__ for component in ACTOR_FRAME_COMPONENTS],
        "actor_frame_dimension": actor_frame,
        "actor_history_frames": ACTOR_HISTORY_FRAMES,
        "actor_dimension": actor_frame * ACTOR_HISTORY_FRAMES,
        "critic_privileged_components": [
            component.__dict__ for component in CRITIC_PRIVILEGED_COMPONENTS
        ],
        "critic_privileged_append_dimension": privileged,
        "critic_dimension": actor_frame * ACTOR_HISTORY_FRAMES + privileged,
    }


def formal_gate_snapshot(
    attach: Mapping[str, Any], legacy_training: Mapping[str, Any]
) -> dict[str, Any]:
    """Extract formal S2-03 gates without interpreting development weights."""

    motion = attach["motion"]
    contact = attach["contact_gate"]
    acceptance = attach["acceptance"]
    legacy_termination = legacy_training["termination_contract"]
    return {
        "bilateral_contact_threshold_n": [
            contact["left_contact_force_threshold_n"],
            contact["right_contact_force_threshold_n"],
        ],
        "bilateral_verification_steps": motion["bilateral_verification_steps"],
        "attached_hold_steps": motion["attached_hold_steps"],
        "maximum_episode_steps": legacy_termination["maximum_episode_steps"],
        "auto_reset_within_evaluation_episode": legacy_termination[
            "auto_reset_within_evaluation_episode"
        ],
        "contact_loss_grace_steps": motion["contact_loss_grace_steps"],
        "single_hand_maximum_steps": motion["single_hand_maximum_steps"],
        "per_palm_force_peak_threshold_n": contact["per_palm_force_peak_threshold_n"],
        "per_palm_impulse_threshold_ns": contact["per_palm_impulse_threshold_ns"],
        "excessive_combined_impulse_threshold_ns": contact[
            "excessive_combined_impulse_threshold_ns"
        ],
        "contact_force_rate_threshold_nps": contact["contact_force_rate_threshold_nps"],
        "maximum_box_linear_speed_mps": acceptance["maximum_box_linear_speed_mps"],
        "maximum_box_angular_speed_radps": acceptance["maximum_box_angular_speed_radps"],
        "maximum_box_translation_m": acceptance["maximum_box_translation_m"],
        "maximum_box_yaw_change_rad": acceptance["maximum_box_yaw_change_rad"],
        "maximum_base_xy_excursion_m": acceptance["maximum_base_xy_excursion_m"],
        "minimum_root_height_m": acceptance["minimum_root_height_m"],
        "maximum_root_height_m": acceptance["maximum_root_height_m"],
        "maximum_root_tilt_deg": acceptance["maximum_root_tilt_deg"],
        "minimum_arm_joint_limit_margin_rad": acceptance["minimum_arm_joint_limit_margin_rad"],
        "maximum_arm_torque_ratio": acceptance["maximum_arm_torque_ratio"],
        "post_initial_reset_count_limit": acceptance["post_initial_reset_count_limit"],
        "pushing_enabled": False,
        "planner_enabled": False,
    }


def validate_contract_bundle(
    action: Mapping[str, Any],
    reward: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    attach: Mapping[str, Any],
    legacy_training: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all cross-file invariants and return the frozen gate snapshot."""

    if any(config.get("stage") != STAGE for config in (action, reward, curriculum)):
        raise ContractError("all redesign contracts must be S2-03T")
    if any(
        config.get("status") != "IMPLEMENTED_FOR_NEW_TRAINING"
        for config in (action, reward, curriculum)
    ):
        raise ContractError("all redesign contracts must be implemented for new training")
    actor_action = action["actor_action"]
    if actor_action["dimension"] != ACTION_DIM or tuple(actor_action["order"]) != ACTION_ORDER:
        raise ContractError("2-D left/right action order mismatch")
    if actor_action["normalized_range"] != [ACTION_MIN, ACTION_MAX]:
        raise ContractError("normalized action range mismatch")
    if not math.isclose(actor_action["correction_scale_m"], CORRECTION_SCALE_M):
        raise ContractError("correction scale mismatch")
    if not math.isclose(
        actor_action["correction_rate_limit_m_per_control_step"],
        CORRECTION_RATE_LIMIT_M_PER_CONTROL_STEP,
    ):
        raise ContractError("correction rate limit mismatch")
    if action["source_facts"]["control_dt_s"] != attach["controller"]["control_dt_s"]:
        raise ContractError("control dt differs from frozen S2-03")
    if (
        action["source_facts"]["frozen_s2_02_orientation_wxyz"]
        != attach["precontact"]["desired_palm_quaternion_in_object_wxyz"]
    ):
        raise ContractError("certified palm orientation changed")
    nominal = action["nominal_approach"]
    for redesigned_key, attach_key in (
        ("speed_limit_mps", "approach_speed_mps"),
        ("acceleration_limit_mps2", "approach_acceleration_limit_mps2"),
        ("jerk_limit_mps3", "approach_jerk_limit_mps3"),
    ):
        if not math.isclose(nominal[redesigned_key], attach["motion"][attach_key]):
            raise ContractError(f"nominal {redesigned_key} changed from S2-03")
    if nominal["maximum_displacement_m"] < attach["precontact"]["precontact_gap_m"]:
        raise ContractError("nominal approach cannot span the formal precontact gap")
    safety = action["safety_stop"]["unchanged_thresholds"]
    gate = formal_gate_snapshot(attach, legacy_training)
    safety_mapping = {
        "per_palm_force_peak_n": "per_palm_force_peak_threshold_n",
        "per_palm_impulse_ns": "per_palm_impulse_threshold_ns",
        "combined_impulse_ns": "excessive_combined_impulse_threshold_ns",
        "contact_force_rate_nps": "contact_force_rate_threshold_nps",
        "maximum_box_linear_speed_mps": "maximum_box_linear_speed_mps",
        "maximum_box_angular_speed_radps": "maximum_box_angular_speed_radps",
        "maximum_box_translation_m": "maximum_box_translation_m",
        "maximum_box_yaw_change_rad": "maximum_box_yaw_change_rad",
        "maximum_root_tilt_deg": "maximum_root_tilt_deg",
        "minimum_joint_margin_rad": "minimum_arm_joint_limit_margin_rad",
        "maximum_torque_ratio": "maximum_arm_torque_ratio",
    }
    if any(safety[source] != gate[target] for source, target in safety_mapping.items()):
        raise ContractError("redesign safety threshold differs from formal S2-03")
    if list(reward["modes"]) != [mode.value for mode in RewardMode]:
        raise ContractError("reward mode order mismatch")
    if reward["control_dt_s"] != CONTROL_DT_S:
        raise ContractError("reward dt mismatch")
    force_band = reward["force_band"]
    if [
        force_band["reliable_contact_minimum_n"],
        force_band["soft_force_maximum_n"],
        force_band["hard_force_threshold_n"],
    ] != [FORCE_MINIMUM_N, FORCE_SOFT_MAXIMUM_N, gate["per_palm_force_peak_threshold_n"]]:
        raise ContractError("force band or frozen hard threshold mismatch")
    expected_terms = [(spec.name, spec.weight) for spec in REWARD_TERM_SPECS]
    actual_terms = [(item["name"], float(item["weight"])) for item in reward["terms"]]
    if actual_terms != expected_terms:
        raise ContractError("reward term names or weights mismatch")
    frozen_reward_gates = reward["frozen_scientific_gates"]
    if frozen_reward_gates["thresholds_changed"] is not False:
        raise ContractError("formal reward thresholds may not change")
    if frozen_reward_gates["verify_steps"] != gate["bilateral_verification_steps"]:
        raise ContractError("verify steps changed")
    if frozen_reward_gates["hold_steps"] != gate["attached_hold_steps"]:
        raise ContractError("hold steps changed")
    if frozen_reward_gates["maximum_episode_steps"] != gate["maximum_episode_steps"]:
        raise ContractError("maximum episode steps changed")
    levels = curriculum["levels"]
    actual_levels = [
        (item["id"], item["name"], item["minimum_gap_m"], item["maximum_gap_m"]) for item in levels
    ]
    expected_levels = [
        (item.level_id, item.name, item.minimum_gap_m, item.maximum_gap_m)
        for item in CURRICULUM_LEVELS
    ]
    if actual_levels != expected_levels:
        raise ContractError("curriculum order or gap interval mismatch")
    sampling = curriculum["sampling"]
    if (
        sampling["type"] != "uniform_per_episode_with_full_gap_endpoint_mixture"
        or sampling["full_gap_endpoint_m"] != 0.060
        or sampling["full_gap_endpoint_probability_at_curriculum_3"] != 0.05
    ):
        raise ContractError("curriculum full-gap endpoint sampling mismatch")
    isolation = curriculum["observation_isolation"]
    expected_dimensions = {
        "actor_frame_dimension": ACTOR_FRAME_DIM,
        "actor_history_frames": ACTOR_HISTORY_FRAMES,
        "actor_dimension": ACTOR_DIM,
        "critic_privileged_append_dimension": CRITIC_PRIVILEGED_APPEND_DIM,
        "critic_dimension": CRITIC_DIM,
    }
    if any(isolation[key] != value for key, value in expected_dimensions.items()):
        raise ContractError("actor/critic observation dimensions mismatch")
    if any(
        isolation[key]
        for key in ("exact_force_in_actor", "jacobian_authority_in_actor", "torque_margin_in_actor")
    ):
        raise ContractError("privileged simulator signals leaked into actor")
    if not all(
        isolation[key]
        for key in (
            "exact_force_in_critic",
            "jacobian_authority_in_critic",
            "torque_margin_in_critic",
        )
    ):
        raise ContractError("critic privileged observation is incomplete")
    boundaries = {**action["frozen_boundaries"], **curriculum["frozen_boundaries"]}
    forbidden_true = (
        "pushing_enabled",
        "planner_enabled",
        "falcon_enabled",
        "old_checkpoint_resume_enabled",
        "active_controller_manifest_update_allowed",
    )
    if any(boundaries.get(key, False) for key in forbidden_true):
        raise ContractError("a frozen project boundary was enabled")
    return gate


def _step(**updates: Any) -> RewardStepInput:
    base = RewardStepInput(mode=RewardMode.APPROACH, surface_gap_m=(0.025, 0.025))
    return replace(base, **updates)


def reward_landscape_v2() -> dict[str, Any]:
    """Return deterministic state rewards, episode returns, and relation checks."""

    state_inputs = {
        "gap_60_mm": _step(surface_gap_m=(0.060, 0.060)),
        "gap_30_mm": _step(surface_gap_m=(0.030, 0.030)),
        "gap_25_mm": _step(surface_gap_m=(0.025, 0.025)),
        "gap_10_mm": _step(surface_gap_m=(0.010, 0.010)),
        "gap_5_mm": _step(surface_gap_m=(0.005, 0.005)),
        "single_hand_contact": _step(
            mode=RewardMode.CONTACT_ACQUIRE,
            left_contact=True,
            left_contact_onset=True,
            palm_force_n=(2.0, 0.0),
        ),
        "bilateral_contact_onset": _step(
            mode=RewardMode.CONTACT_ACQUIRE,
            left_contact=True,
            right_contact=True,
            left_contact_onset=True,
            right_contact_onset=True,
            bilateral_contact_onset=True,
            verify_progress_fraction=1.0 / VERIFY_STEPS,
            palm_force_n=(2.0, 2.0),
        ),
        "bilateral_verify": _step(
            mode=RewardMode.CONTACT_ACQUIRE,
            left_contact=True,
            right_contact=True,
            verify_progress_fraction=1.0 / VERIFY_STEPS,
            palm_force_n=(2.0, 2.0),
        ),
        "safe_hold": _step(
            mode=RewardMode.VERIFY_HOLD,
            left_contact=True,
            right_contact=True,
            palm_force_n=(2.0, 2.0),
        ),
        "hard_impact": _step(
            mode=RewardMode.CONTACT_ACQUIRE,
            left_contact=True,
            right_contact=True,
            palm_force_n=(FORCE_HARD_THRESHOLD_N, FORCE_HARD_THRESHOLD_N),
            hard_force_terminal=True,
        ),
        "pushing": _step(
            mode=RewardMode.VERIFY_HOLD,
            left_contact=True,
            right_contact=True,
            palm_force_n=(2.0, 2.0),
            pushing_terminal=True,
        ),
        "timeout": _step(contact_timeout_terminal=True),
    }
    states = {
        name: {
            "total": breakdown.total,
            "raw": breakdown.raw,
            "actual_contribution": breakdown.actual_contribution,
        }
        for name, breakdown in (
            (name, compute_reward_step(step)) for name, step in state_inputs.items()
        )
    }

    hover = compute_reward_step(state_inputs["gap_25_mm"]).total
    timeout = compute_reward_step(state_inputs["timeout"]).total
    hover_return = hover * (MAXIMUM_EPISODE_STEPS - 1) + timeout

    approach_progress = compute_reward_step(
        _step(surface_gap_m=(0.0595, 0.0595), gap_progress_m=PROGRESS_CLIP_M)
    ).total
    approach_then_hover_return = (
        approach_progress * 70 + hover * (MAXIMUM_EPISODE_STEPS - 71) + timeout
    )

    onset = compute_reward_step(state_inputs["bilateral_contact_onset"]).total
    verify = compute_reward_step(state_inputs["bilateral_verify"]).total
    hold = compute_reward_step(state_inputs["safe_hold"]).total
    success = compute_reward_step(
        replace(state_inputs["safe_hold"], frozen_attach_success=True)
    ).total
    safe_hold_return = onset + verify * (VERIFY_STEPS - 1) + hold * (HOLD_STEPS - 1) + success

    single_first = compute_reward_step(state_inputs["single_hand_contact"]).total
    single_regular = compute_reward_step(
        replace(state_inputs["single_hand_contact"], left_contact_onset=False)
    ).total
    single_terminal = compute_reward_step(
        replace(
            state_inputs["single_hand_contact"],
            left_contact_onset=False,
            contact_timeout_terminal=True,
        )
    ).total
    single_timeout_return = single_first + single_regular * 23 + single_terminal
    hard_impact_return = compute_reward_step(state_inputs["hard_impact"]).total
    pushing_return = compute_reward_step(state_inputs["pushing"]).total
    episodes = {
        "hover_25_mm_1000_steps": hover_return,
        "gap_60_to_25_then_hover": approach_then_hover_return,
        "bilateral_contact_verify_safe_hold": safe_hold_return,
        "single_hand_contact_then_timeout": single_timeout_return,
        "hard_impact_immediate": hard_impact_return,
        "contact_then_pushing": pushing_return,
    }
    relations = {
        "safe_hold_return_gt_hover_25_mm_return": safe_hold_return > hover_return,
        "safe_hold_return_gt_single_contact_timeout_return": safe_hold_return
        > single_timeout_return,
        "hard_impact_return_lt_safe_hold_return": hard_impact_return < safe_hold_return,
        "pushing_return_lt_safe_hold_return": pushing_return < safe_hold_return,
        "hover_25_mm_1000_step_return_lte_zero": hover_return <= 0.0,
    }
    return {"states": states, "episode_returns": episodes, "required_relations": relations}


def reward_term_budget() -> list[dict[str, Any]]:
    """Return units and conservative per-step/episode contribution bounds."""

    event_names = {spec.name for spec in REWARD_TERM_SPECS if spec.event_once_per_episode}
    records: list[dict[str, Any]] = []
    for spec in REWARD_TERM_SPECS:
        maximum_episode: float | str
        if isinstance(spec.maximum_actual_step_contribution, str):
            maximum_episode = spec.maximum_actual_step_contribution
        elif spec.name in event_names:
            maximum_episode = spec.maximum_actual_step_contribution
        elif spec.name == "verify_progress":
            maximum_episode = spec.maximum_actual_step_contribution * VERIFY_STEPS
        elif spec.name == "attached_hold_step":
            maximum_episode = spec.maximum_actual_step_contribution * HOLD_STEPS
        else:
            maximum_episode = spec.maximum_actual_step_contribution * MAXIMUM_EPISODE_STEPS
        records.append(
            {
                "name": spec.name,
                "weight": spec.weight,
                "raw_unit": spec.raw_unit,
                "enabled_modes": [mode.value for mode in spec.enabled_modes],
                "maximum_actual_step_contribution": spec.maximum_actual_step_contribution,
                "maximum_episode_contribution": maximum_episode,
            }
        )
    return records


def build_resolved_config(
    *,
    action_path: Path,
    reward_path: Path,
    curriculum_path: Path,
    attach_path: Path,
    legacy_training_path: Path,
) -> dict[str, Any]:
    """Build one deterministic, source-hashed redesign artifact."""

    action = load_yaml(action_path)
    reward = load_yaml(reward_path)
    curriculum = load_yaml(curriculum_path)
    attach = load_yaml(attach_path)
    legacy_training = load_yaml(legacy_training_path)
    gates = validate_contract_bundle(action, reward, curriculum, attach, legacy_training)
    landscape = reward_landscape_v2()
    if not all(landscape["required_relations"].values()):
        raise ContractError("redesigned reward return relations failed")
    observation = observation_contract_payload()
    if observation["actor_dimension"] != ACTOR_DIM or observation["critic_dimension"] != CRITIC_DIM:
        raise ContractError("resolved observation dimensions failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": "READY_FOR_RUNTIME_INTEGRATION",
        "development_scope": "SIMULATION_DEVELOPMENT_ONLY",
        "source_sha256": {
            str(action_path): sha256_file(action_path),
            str(reward_path): sha256_file(reward_path),
            str(curriculum_path): sha256_file(curriculum_path),
            str(attach_path): sha256_file(attach_path),
            str(legacy_training_path): sha256_file(legacy_training_path),
        },
        "action": action,
        "reward": reward,
        "curriculum": curriculum,
        "observation": observation,
        "formal_s2_03_gates": gates,
        "formal_s2_03_gates_unchanged": True,
        "reward_landscape_v2": landscape,
        "reward_term_budget": reward_term_budget(),
        "frozen_prohibitions": {
            "old_checkpoint_resume": False,
            "model_1999_used": False,
            "falcon_migration_started": False,
            "box_pushing_enabled": False,
            "planner_started": False,
            "s2_04_started": False,
            "active_controller_manifest_update_allowed": False,
        },
    }


def canonical_json(payload: Any) -> str:
    """Serialize artifacts reproducibly for generation and check mode."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
