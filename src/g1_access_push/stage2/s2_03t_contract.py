"""Pure, dependency-light contracts for S2-03T contact training.

This module deliberately contains no Isaac Lab imports.  Runtime code may use the
constants and helpers below, while static tests can audit the scientific interface
without starting Kit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


TRAINING_CONTRACT_SHA256 = "221f27d2a22d3e07f9bd9a05680d61d57f3b2af59a2cf0ed4f5bd8af424e6316"
OBSERVATION_DIM = 77
ACTION_DIM = 14
LOWER_BODY_ACTION_DIM = 0
RESIDUAL_SCALE_RAD = 0.05
MAXIMUM_RESIDUAL_CHANGE_RAD = 0.005
JOINT_LIMIT_MARGIN_RAD = 0.10
CONTROL_FREQUENCY_HZ = 50
CONTROL_DT_S = 1.0 / CONTROL_FREQUENCY_HZ


class ContractError(ValueError):
    """Raised when a value cannot satisfy the frozen S2-03T contract."""


@dataclass(frozen=True)
class ObservationComponent:
    name: str
    dimension: int
    unit: str
    clip_min: float
    clip_max: float


OBSERVATION_COMPONENTS = (
    ObservationComponent("base_ang_vel_body", 3, "rad_per_s", -5.0, 5.0),
    ObservationComponent("projected_gravity_body", 3, "dimensionless", -1.0, 1.0),
    ObservationComponent("arm_joint_position_error_to_ik_reference", 14, "rad", -1.0, 1.0),
    ObservationComponent("arm_joint_velocity", 14, "rad_per_s", -10.0, 10.0),
    ObservationComponent("left_palm_position_error_in_box", 3, "m", -0.10, 0.10),
    ObservationComponent("right_palm_position_error_in_box", 3, "m", -0.10, 0.10),
    ObservationComponent("left_palm_orientation_error_axis_angle_in_box", 3, "rad", -0.5, 0.5),
    ObservationComponent("right_palm_orientation_error_axis_angle_in_box", 3, "rad", -0.5, 0.5),
    ObservationComponent("palm_actual_surface_gap", 2, "m", -0.02, 0.10),
    ObservationComponent("palm_box_force_norm", 2, "N", 0.0, 9.81),
    ObservationComponent("palm_contact_flags", 2, "bool", 0.0, 1.0),
    ObservationComponent("box_translation_from_attach_start", 3, "m", -0.005, 0.005),
    ObservationComponent(
        "box_yaw_change_from_attach_start", 1, "rad", -0.008726646259971648, 0.008726646259971648
    ),
    ObservationComponent("box_linear_velocity", 3, "m_per_s", -0.005, 0.005),
    ObservationComponent(
        "box_angular_velocity", 3, "rad_per_s", -0.008726646259971648, 0.008726646259971648
    ),
    ObservationComponent("approach_progress_fraction", 1, "dimensionless", 0.0, 1.0),
    ObservationComponent("previous_arm_action", 14, "normalized", -1.0, 1.0),
)


ACTION_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class RewardSpec:
    name: str
    weight: float
    formula: str


REWARD_SPECS = (
    RewardSpec("bilateral_contact_verify_step", 2.0, "both_contact_flags"),
    RewardSpec("attached_hold_step", 4.0, "both_contacts_and_all_frozen_gates"),
    RewardSpec("success_terminal", 100.0, "frozen_attach_success"),
    RewardSpec("symmetric_gap_closure", 1.0, "exp_minus_50_times_sum_abs_surface_gap"),
    RewardSpec("hand_position_tracking", -2.0, "sum_palm_position_error_m"),
    RewardSpec("hand_orientation_tracking", -0.1, "sum_palm_orientation_error_rad"),
    RewardSpec("force_imbalance", -0.5, "abs_left_force_minus_right_force_n"),
    RewardSpec("box_translation", -40.0, "box_translation_m"),
    RewardSpec("box_yaw_change", -40.0, "abs_box_yaw_change_rad"),
    RewardSpec("force_impulse_rate", -2.0, "normalized_peak_plus_impulse_plus_rate"),
    RewardSpec("root_tilt", -0.1, "root_tilt_deg"),
    RewardSpec("joint_limit_margin", -5.0, "positive_part_0p10_minus_margin_rad"),
    RewardSpec("torque_ratio", -2.0, "positive_part_torque_ratio_minus_1"),
    RewardSpec("action_l2", -0.001, "squared_action_norm"),
    RewardSpec("action_rate", -0.01, "squared_action_delta_norm"),
    RewardSpec("forbidden_collision_terminal", -100.0, "any_non_palm_box_collision"),
    RewardSpec("contact_timeout_terminal", -25.0, "bilateral_contact_timeout"),
)


TERMINATION_LIMITS: dict[str, float | int] = {
    "bilateral_contact_threshold_n": 1.0,
    "bilateral_verification_steps": 20,
    "attached_hold_steps": 100,
    "maximum_episode_steps": 1000,
    "contact_loss_grace_steps": 5,
    "single_hand_maximum_steps": 25,
    "per_palm_force_peak_threshold_n": 9.81,
    "per_palm_impulse_threshold_ns": 0.981,
    "excessive_combined_impulse_threshold_ns": 1.962,
    "contact_force_rate_threshold_nps": 490.5,
    "maximum_box_linear_speed_mps": 0.005,
    "maximum_box_angular_speed_radps": 0.008726646259971648,
    "maximum_box_translation_m": 0.005,
    "maximum_box_yaw_change_rad": 0.008726646259971648,
    "maximum_base_xy_excursion_m": 0.05,
    "minimum_root_height_m": 0.5,
    "maximum_root_height_m": 1.0,
    "maximum_root_tilt_deg": 6.2075676918029785,
    "minimum_arm_joint_limit_margin_rad": 0.10,
    "maximum_arm_torque_ratio": 1.001,
}


TERMINATION_SUCCESS_CONTRACT: dict[str, Any] = {
    "bilateral_contact_threshold_n": [1.0, 1.0],
    "bilateral_verification_steps": 20,
    "attached_hold_steps": 100,
    "all_frozen_safety_gates_required": True,
}

TERMINATION_FAILURE_CONTRACT: dict[str, Any] = {
    "nonfinite": "immediate",
    "forbidden_non_palm_box_collision": "immediate",
    "bilateral_contact_timeout": 1000,
    "contact_loss_grace_steps": 5,
    "single_hand_maximum_steps": 25,
    "per_palm_force_peak_threshold_n": 9.81,
    "per_palm_impulse_threshold_ns": 0.981,
    "excessive_combined_impulse_threshold_ns": 1.962,
    "contact_force_rate_threshold_nps": 490.5,
    "maximum_box_linear_speed_mps": 0.005,
    "maximum_box_angular_speed_radps": 0.008726646259971648,
    "maximum_box_translation_m": 0.005,
    "maximum_box_yaw_change_rad": 0.008726646259971648,
    "maximum_base_xy_excursion_m": 0.05,
    "minimum_root_height_m": 0.5,
    "maximum_root_height_m": 1.0,
    "maximum_root_tilt_deg": 6.2075676918029785,
    "minimum_arm_joint_limit_margin_rad": 0.10,
    "maximum_arm_torque_ratio": 1.001,
    "post_initial_reset_count_limit": 0,
}


PHYSICAL_TERMINATION_REASONS = (
    "NONFINITE",
    "FORBIDDEN_NON_PALM_BOX_COLLISION",
    "LEFT_FORCE_PEAK",
    "RIGHT_FORCE_PEAK",
    "LEFT_PALM_IMPULSE",
    "RIGHT_PALM_IMPULSE",
    "EXCESSIVE_COMBINED_IMPULSE",
    "CONTACT_FORCE_RATE",
    "BOX_LINEAR_SPEED",
    "BOX_ANGULAR_SPEED",
    "BOX_TRANSLATION",
    "BOX_YAW_CHANGE",
    "BASE_XY_EXCURSION",
    "ROOT_HEIGHT",
    "ROOT_TILT",
    "ARM_JOINT_MARGIN",
    "ARM_TORQUE_RATIO",
    "CONTACT_LOSS",
    "SINGLE_HAND_TIMEOUT",
)


TRACKER_SIGNAL_FIELDS = (
    "left_force_n",
    "right_force_n",
    "finite",
    "forbidden_non_palm_box_collision",
    "all_frozen_safety_gates",
    "box_linear_speed_mps",
    "box_angular_speed_radps",
    "box_translation_m",
    "box_yaw_change_rad",
    "base_xy_excursion_m",
    "root_height_m",
    "root_tilt_deg",
    "minimum_arm_joint_limit_margin_rad",
    "arm_torque_ratio_max",
)


def observation_slices() -> dict[str, slice]:
    """Return the frozen, insertion-ordered slices for the 77-D observation."""

    result: dict[str, slice] = {}
    start = 0
    for component in OBSERVATION_COMPONENTS:
        result[component.name] = slice(start, start + component.dimension)
        start += component.dimension
    if start != OBSERVATION_DIM:
        raise AssertionError(f"internal observation layout is {start}, expected {OBSERVATION_DIM}")
    return result


def observation_layout_payload() -> dict[str, Any]:
    """Return a machine-readable layout with zero-based, half-open ranges."""

    slices = observation_slices()
    return {
        "schema_version": 1,
        "stage": "S2-03T",
        "policy_observation_dim": OBSERVATION_DIM,
        "batch_first": True,
        "representation": "RAW_SI_WITH_COMPONENT_CLIPPING",
        "finite_required": True,
        "index_convention": "zero_based_half_open",
        "components": [
            {
                "order": order,
                "name": component.name,
                "dimension": component.dimension,
                "start": slices[component.name].start,
                "stop": slices[component.name].stop,
                "inclusive_index_range": [slices[component.name].start, slices[component.name].stop - 1],
                "unit": component.unit,
                "clip_min": component.clip_min,
                "clip_max": component.clip_max,
            }
            for order, component in enumerate(OBSERVATION_COMPONENTS, start=1)
        ],
    }


def _finite_float(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be numeric") from exc
    if not math.isfinite(converted):
        raise ContractError(f"{label} must be finite")
    return converted


def assemble_observation(components: Mapping[str, Sequence[Sequence[float]]]) -> list[list[float]]:
    """Clip and concatenate component matrices into a batch-first 77-D list.

    Every component is required and must have shape ``(batch, component_dim)``.
    Values are checked for finiteness before clipping, so NaN/Inf cannot be hidden.
    """

    expected = {component.name for component in OBSERVATION_COMPONENTS}
    supplied = set(components)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ContractError(f"observation components mismatch: missing={missing}, extra={extra}")
    batch_size: int | None = None
    output: list[list[float]] = []
    for component in OBSERVATION_COMPONENTS:
        rows = components[component.name]
        if batch_size is None:
            batch_size = len(rows)
            if batch_size < 1:
                raise ContractError("observation batch must not be empty")
            output = [[] for _ in range(batch_size)]
        elif len(rows) != batch_size:
            raise ContractError(f"{component.name} has inconsistent batch size")
        for batch_index, row in enumerate(rows):
            if len(row) != component.dimension:
                raise ContractError(
                    f"{component.name}[{batch_index}] has dimension {len(row)}, expected {component.dimension}"
                )
            output[batch_index].extend(
                min(component.clip_max, max(component.clip_min, _finite_float(value, component.name)))
                for value in row
            )
    if any(len(row) != OBSERVATION_DIM for row in output):
        raise AssertionError("internal observation concatenation error")
    return output


def _vector(values: Sequence[float], name: str, expected: int = ACTION_DIM) -> list[float]:
    if len(values) != expected:
        raise ContractError(f"{name} has dimension {len(values)}, expected {expected}")
    return [_finite_float(value, f"{name}[{index}]") for index, value in enumerate(values)]


@dataclass(frozen=True)
class ArmActionMapping:
    clipped_action: tuple[float, ...]
    requested_residual_rad: tuple[float, ...]
    applied_residual_rad: tuple[float, ...]
    joint_target_rad: tuple[float, ...]
    minimum_joint_limit_margin_rad: float
    joint_limit_clamped: tuple[bool, ...]


def map_arm_residual_action(
    action: Sequence[float],
    previous_residual_rad: Sequence[float],
    frozen_ik_reference_rad: Sequence[float],
    hard_lower_limits_rad: Sequence[float],
    hard_upper_limits_rad: Sequence[float],
) -> ArmActionMapping:
    """Map one normalized 14-D policy action to safe arm joint targets."""

    action_v = _vector(action, "action")
    previous_v = _vector(previous_residual_rad, "previous_residual_rad")
    reference_v = _vector(frozen_ik_reference_rad, "frozen_ik_reference_rad")
    lower_v = _vector(hard_lower_limits_rad, "hard_lower_limits_rad")
    upper_v = _vector(hard_upper_limits_rad, "hard_upper_limits_rad")
    clipped: list[float] = []
    requested: list[float] = []
    applied: list[float] = []
    targets: list[float] = []
    clamped: list[bool] = []
    margins: list[float] = []
    for index in range(ACTION_DIM):
        safe_lower = lower_v[index] + JOINT_LIMIT_MARGIN_RAD
        safe_upper = upper_v[index] - JOINT_LIMIT_MARGIN_RAD
        if safe_lower > safe_upper:
            raise ContractError(f"joint {ACTION_JOINT_NAMES[index]} has no {JOINT_LIMIT_MARGIN_RAD} rad safe range")
        if not safe_lower <= reference_v[index] <= safe_upper:
            raise ContractError(f"frozen IK reference violates margin for {ACTION_JOINT_NAMES[index]}")
        normalized = min(1.0, max(-1.0, action_v[index]))
        desired_residual = normalized * RESIDUAL_SCALE_RAD
        delta = min(
            MAXIMUM_RESIDUAL_CHANGE_RAD,
            max(-MAXIMUM_RESIDUAL_CHANGE_RAD, desired_residual - previous_v[index]),
        )
        rate_limited_residual = previous_v[index] + delta
        raw_target = reference_v[index] + rate_limited_residual
        target = min(safe_upper, max(safe_lower, raw_target))
        clipped.append(normalized)
        requested.append(desired_residual)
        applied.append(target - reference_v[index])
        targets.append(target)
        clamped.append(not math.isclose(target, raw_target, rel_tol=0.0, abs_tol=1.0e-12))
        margins.append(min(target - lower_v[index], upper_v[index] - target))
    return ArmActionMapping(
        clipped_action=tuple(clipped),
        requested_residual_rad=tuple(requested),
        applied_residual_rad=tuple(applied),
        joint_target_rad=tuple(targets),
        minimum_joint_limit_margin_rad=min(margins),
        joint_limit_clamped=tuple(clamped),
    )


def _pair(metrics: Mapping[str, Any], name: str) -> tuple[float, float]:
    values = metrics.get(name)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
        raise ContractError(f"reward metric {name} must contain left/right values")
    return _finite_float(values[0], f"{name}[0]"), _finite_float(values[1], f"{name}[1]")


def compute_reward_terms(
    metrics: Mapping[str, Any], action: Sequence[float], previous_action: Sequence[float]
) -> dict[str, dict[str, float] | float]:
    """Evaluate every frozen reward formula and its weighted contribution."""

    action_v = _vector(action, "action")
    previous_v = _vector(previous_action, "previous_action")
    gaps = _pair(metrics, "surface_gap_m")
    position_errors = _pair(metrics, "palm_position_error_m")
    orientation_errors = _pair(metrics, "palm_orientation_error_rad")
    forces = _pair(metrics, "palm_force_norm_n")
    left_contact = bool(metrics.get("left_contact", False))
    right_contact = bool(metrics.get("right_contact", False))
    both = left_contact and right_contact
    raw = {
        "bilateral_contact_verify_step": float(both),
        "attached_hold_step": float(both and bool(metrics.get("all_frozen_gates", False))),
        "success_terminal": float(bool(metrics.get("frozen_attach_success", False))),
        "symmetric_gap_closure": math.exp(-50.0 * (abs(gaps[0]) + abs(gaps[1]))),
        "hand_position_tracking": position_errors[0] + position_errors[1],
        "hand_orientation_tracking": orientation_errors[0] + orientation_errors[1],
        "force_imbalance": abs(forces[0] - forces[1]),
        "box_translation": _finite_float(metrics.get("box_translation_m"), "box_translation_m"),
        "box_yaw_change": abs(_finite_float(metrics.get("box_yaw_change_rad"), "box_yaw_change_rad")),
        "force_impulse_rate": _finite_float(
            metrics.get("normalized_peak_plus_impulse_plus_rate"), "normalized_peak_plus_impulse_plus_rate"
        ),
        "root_tilt": _finite_float(metrics.get("root_tilt_deg"), "root_tilt_deg"),
        "joint_limit_margin": max(
            0.0,
            JOINT_LIMIT_MARGIN_RAD
            - _finite_float(metrics.get("minimum_arm_joint_limit_margin_rad"), "minimum_arm_joint_limit_margin_rad"),
        ),
        "torque_ratio": max(0.0, _finite_float(metrics.get("arm_torque_ratio_max"), "arm_torque_ratio_max") - 1.0),
        "action_l2": sum(value * value for value in action_v),
        "action_rate": sum((value - previous) ** 2 for value, previous in zip(action_v, previous_v, strict=True)),
        "forbidden_collision_terminal": float(bool(metrics.get("forbidden_non_palm_box_collision", False))),
        "contact_timeout_terminal": float(bool(metrics.get("bilateral_contact_timeout", False))),
    }
    weights = {spec.name: spec.weight for spec in REWARD_SPECS}
    weighted = {name: weights[name] * value for name, value in raw.items()}
    return {"raw": raw, "weighted": weighted, "total": sum(weighted.values())}


def reward_contract_records() -> list[dict[str, Any]]:
    return [{"name": spec.name, "weight": spec.weight, "formula": spec.formula} for spec in REWARD_SPECS]


@dataclass(frozen=True)
class TerminationSnapshot:
    step: int
    terminated: bool
    truncated: bool
    time_out: bool
    success: bool
    bilateral_contact_timeout: bool
    reasons: tuple[str, ...]
    extras: dict[str, bool]


class AttachEpisodeTracker:
    """Stateful contact/impulse/FSM counters with separated truncation semantics."""

    def __init__(self) -> None:
        self.post_initial_reset_count = 0
        self.reset(initial=True)

    def reset(self, *, initial: bool = False) -> None:
        if not initial:
            self.post_initial_reset_count += 1
        self.step_count = 0
        self.bilateral_verify_steps = 0
        self.attached_hold_steps = 0
        self.contact_loss_steps = 0
        self.single_hand_steps = 0
        self.bilateral_contact_ever = False
        self.bilateral_verified = False
        self.previous_force_n = [0.0, 0.0]
        self._left_impulse_window: deque[float] = deque(maxlen=5)
        self._right_impulse_window: deque[float] = deque(maxlen=5)
        self.left_contact = False
        self.right_contact = False
        self.success = False
        self.last_snapshot: TerminationSnapshot | None = None

    @property
    def left_impulse_ns(self) -> float:
        return sum(self._left_impulse_window)

    @property
    def right_impulse_ns(self) -> float:
        return sum(self._right_impulse_window)

    def step(self, signals: Mapping[str, Any]) -> TerminationSnapshot:
        missing = sorted(set(TRACKER_SIGNAL_FIELDS) - set(signals))
        if missing:
            raise ContractError(f"tracker signals missing fields: {missing}")
        self.step_count += 1
        finite_flag = bool(signals["finite"])
        numeric: dict[str, float] = {}
        for name in TRACKER_SIGNAL_FIELDS:
            if name in {"finite", "forbidden_non_palm_box_collision", "all_frozen_safety_gates"}:
                continue
            try:
                numeric[name] = float(signals[name])
            except (TypeError, ValueError):
                numeric[name] = math.nan
            finite_flag = finite_flag and math.isfinite(numeric[name])

        left_force = numeric["left_force_n"] if math.isfinite(numeric["left_force_n"]) else 0.0
        right_force = numeric["right_force_n"] if math.isfinite(numeric["right_force_n"]) else 0.0
        self.left_contact = left_force >= float(TERMINATION_LIMITS["bilateral_contact_threshold_n"])
        self.right_contact = right_force >= float(TERMINATION_LIMITS["bilateral_contact_threshold_n"])
        both = self.left_contact and self.right_contact
        if both:
            self.bilateral_contact_ever = True
        self._left_impulse_window.append(abs(left_force) * CONTROL_DT_S)
        self._right_impulse_window.append(abs(right_force) * CONTROL_DT_S)
        force_rate = max(
            abs(left_force - self.previous_force_n[0]) / CONTROL_DT_S,
            abs(right_force - self.previous_force_n[1]) / CONTROL_DT_S,
        )
        self.previous_force_n[:] = [left_force, right_force]

        if not self.bilateral_verified:
            self.bilateral_verify_steps = self.bilateral_verify_steps + 1 if both else 0
            if self.bilateral_verify_steps >= int(TERMINATION_LIMITS["bilateral_verification_steps"]):
                self.bilateral_verified = True
        elif both and bool(signals["all_frozen_safety_gates"]):
            self.attached_hold_steps += 1
        else:
            self.attached_hold_steps = 0

        if self.bilateral_contact_ever and not both:
            self.contact_loss_steps += 1
        else:
            self.contact_loss_steps = 0
        self.single_hand_steps = self.single_hand_steps + 1 if self.left_contact ^ self.right_contact else 0

        reasons: list[str] = []
        if not finite_flag:
            reasons.append("NONFINITE")
        if bool(signals["forbidden_non_palm_box_collision"]):
            reasons.append("FORBIDDEN_NON_PALM_BOX_COLLISION")
        if left_force > float(TERMINATION_LIMITS["per_palm_force_peak_threshold_n"]):
            reasons.append("LEFT_FORCE_PEAK")
        if right_force > float(TERMINATION_LIMITS["per_palm_force_peak_threshold_n"]):
            reasons.append("RIGHT_FORCE_PEAK")
        if self.left_impulse_ns > float(TERMINATION_LIMITS["per_palm_impulse_threshold_ns"]):
            reasons.append("LEFT_PALM_IMPULSE")
        if self.right_impulse_ns > float(TERMINATION_LIMITS["per_palm_impulse_threshold_ns"]):
            reasons.append("RIGHT_PALM_IMPULSE")
        if self.left_impulse_ns + self.right_impulse_ns > float(
            TERMINATION_LIMITS["excessive_combined_impulse_threshold_ns"]
        ):
            reasons.append("EXCESSIVE_COMBINED_IMPULSE")
        if force_rate > float(TERMINATION_LIMITS["contact_force_rate_threshold_nps"]):
            reasons.append("CONTACT_FORCE_RATE")
        gate_checks = (
            ("BOX_LINEAR_SPEED", numeric["box_linear_speed_mps"], "maximum_box_linear_speed_mps"),
            ("BOX_ANGULAR_SPEED", numeric["box_angular_speed_radps"], "maximum_box_angular_speed_radps"),
            ("BOX_TRANSLATION", numeric["box_translation_m"], "maximum_box_translation_m"),
            ("BOX_YAW_CHANGE", numeric["box_yaw_change_rad"], "maximum_box_yaw_change_rad"),
            ("BASE_XY_EXCURSION", numeric["base_xy_excursion_m"], "maximum_base_xy_excursion_m"),
            ("ROOT_TILT", numeric["root_tilt_deg"], "maximum_root_tilt_deg"),
            ("ARM_TORQUE_RATIO", numeric["arm_torque_ratio_max"], "maximum_arm_torque_ratio"),
        )
        for reason, value, limit_name in gate_checks:
            if math.isfinite(value) and abs(value) > float(TERMINATION_LIMITS[limit_name]):
                reasons.append(reason)
        root_height = numeric["root_height_m"]
        if math.isfinite(root_height) and not (
            float(TERMINATION_LIMITS["minimum_root_height_m"])
            <= root_height
            <= float(TERMINATION_LIMITS["maximum_root_height_m"])
        ):
            reasons.append("ROOT_HEIGHT")
        margin = numeric["minimum_arm_joint_limit_margin_rad"]
        if math.isfinite(margin) and margin < float(TERMINATION_LIMITS["minimum_arm_joint_limit_margin_rad"]):
            reasons.append("ARM_JOINT_MARGIN")
        if self.contact_loss_steps > int(TERMINATION_LIMITS["contact_loss_grace_steps"]):
            reasons.append("CONTACT_LOSS")
        if self.single_hand_steps > int(TERMINATION_LIMITS["single_hand_maximum_steps"]):
            reasons.append("SINGLE_HAND_TIMEOUT")

        reasons = list(dict.fromkeys(reasons))
        terminated = bool(reasons)
        self.success = (
            not terminated
            and self.bilateral_verified
            and self.attached_hold_steps >= int(TERMINATION_LIMITS["attached_hold_steps"])
        )
        time_out = self.step_count >= int(TERMINATION_LIMITS["maximum_episode_steps"]) and not self.success
        bilateral_contact_timeout = time_out
        extras = {reason: reason in reasons for reason in PHYSICAL_TERMINATION_REASONS}
        extras.update(
            {
                "success": self.success,
                "time_out": time_out,
                "bilateral_contact_timeout": bilateral_contact_timeout,
                "left_contact": self.left_contact,
                "right_contact": self.right_contact,
            }
        )
        snapshot = TerminationSnapshot(
            step=self.step_count,
            terminated=terminated,
            truncated=time_out,
            time_out=time_out,
            success=self.success,
            bilateral_contact_timeout=bilateral_contact_timeout,
            reasons=tuple(reasons),
            extras=extras,
        )
        self.last_snapshot = snapshot
        return snapshot

    def state(self) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "left_contact": self.left_contact,
            "right_contact": self.right_contact,
            "bilateral_contact_ever": self.bilateral_contact_ever,
            "bilateral_verified": self.bilateral_verified,
            "bilateral_verify_steps": self.bilateral_verify_steps,
            "attached_hold_steps": self.attached_hold_steps,
            "contact_loss_steps": self.contact_loss_steps,
            "single_hand_steps": self.single_hand_steps,
            "left_impulse_ns": self.left_impulse_ns,
            "right_impulse_ns": self.right_impulse_ns,
            "combined_impulse_ns": self.left_impulse_ns + self.right_impulse_ns,
            "previous_force_n": list(self.previous_force_n),
            "post_initial_reset_count": self.post_initial_reset_count,
            "success": self.success,
        }


RESULT_TYPES = {
    "contract_smoke",
    "pilot_baseline",
    "pilot_block",
    "formal_training",
    "checkpoint_screening",
    "formal_qualification",
}


def validate_result_schema(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors; an empty list means the small result is valid."""

    errors: list[str] = []
    required = ("schema_version", "stage", "result_type", "status", "primary_reason", "training_contract_sha256")
    for name in required:
        if name not in payload:
            errors.append(f"missing:{name}")
    if payload.get("schema_version") != 1:
        errors.append("schema_version")
    if payload.get("stage") != "S2-03T":
        errors.append("stage")
    result_type = payload.get("result_type")
    if result_type not in RESULT_TYPES:
        errors.append("result_type")
    if payload.get("status") not in {"PASS", "FAIL", "INVALID"}:
        errors.append("status")
    if not isinstance(payload.get("primary_reason"), str) or not payload.get("primary_reason"):
        errors.append("primary_reason")
    if payload.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256:
        errors.append("training_contract_sha256")
    if result_type == "contract_smoke":
        if payload.get("observation_shape") != [1, OBSERVATION_DIM]:
            errors.append("observation_shape")
        if payload.get("action_shape") != [1, ACTION_DIM]:
            errors.append("action_shape")
        if payload.get("actor_checkpoint_loaded") is not False:
            errors.append("actor_checkpoint_loaded")
        if not isinstance(payload.get("steps"), int) or not 2 <= payload["steps"] <= 5:
            errors.append("steps")
    if result_type == "formal_training":
        if payload.get("clean_actor") is not True:
            errors.append("clean_actor")
        if payload.get("resume") is not False:
            errors.append("resume")
        if payload.get("pilot_checkpoint_used") is not False:
            errors.append("pilot_checkpoint_used")
    if result_type == "formal_qualification":
        if payload.get("episode_count") != 1:
            errors.append("episode_count")
        if payload.get("auto_reset") is not False:
            errors.append("auto_reset")
        if payload.get("post_initial_reset_count") != 0:
            errors.append("post_initial_reset_count")
    return list(dict.fromkeys(errors))


def assert_valid_result_schema(payload: Mapping[str, Any]) -> None:
    errors = validate_result_schema(payload)
    if errors:
        raise ContractError(f"result schema errors: {errors}")
