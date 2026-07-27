"""Attach-only finite-state machine with explicit legal transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contract import INVALID_REASONS, PHYSICAL_FAILURE_REASONS


class AttachState(str, Enum):
    RESET = "RESET"
    STAND_SETTLE = "STAND_SETTLE"
    PRECONTACT = "PRECONTACT"
    APPROACH_NORMAL = "APPROACH_NORMAL"
    BILATERAL_CONTACT_VERIFY = "BILATERAL_CONTACT_VERIFY"
    ATTACHED_HOLD = "ATTACHED_HOLD"
    PASS = "PASS"
    FAIL = "FAIL"


LEGAL_TRANSITIONS = {
    AttachState.RESET: frozenset({AttachState.STAND_SETTLE, AttachState.FAIL}),
    AttachState.STAND_SETTLE: frozenset({AttachState.PRECONTACT, AttachState.FAIL}),
    AttachState.PRECONTACT: frozenset({AttachState.APPROACH_NORMAL, AttachState.FAIL}),
    AttachState.APPROACH_NORMAL: frozenset({AttachState.BILATERAL_CONTACT_VERIFY, AttachState.FAIL}),
    AttachState.BILATERAL_CONTACT_VERIFY: frozenset({AttachState.ATTACHED_HOLD, AttachState.FAIL}),
    AttachState.ATTACHED_HOLD: frozenset({AttachState.PASS, AttachState.FAIL}),
    AttachState.PASS: frozenset(),
    AttachState.FAIL: frozenset(),
}


@dataclass(frozen=True)
class StateMetadata:
    purpose: str
    allowed_commands: tuple[str, ...]
    forbidden_commands: tuple[str, ...]
    required_observations: tuple[str, ...]
    entry_conditions: tuple[str, ...]
    exit_conditions: tuple[str, ...]
    possible_failure_reasons: tuple[str, ...]


_COMMON_FORBIDDEN = ("push", "planner_transition", "automatic_reset_accumulation")
_COMMON_OBS = ("left_contact", "right_contact", "object_position_xyz_m", "root_height_m", "root_tilt_rad")
STATE_METADATA = {
    AttachState.RESET: StateMetadata("initialize one episode", ("reset",), _COMMON_FORBIDDEN, _COMMON_OBS, ("episode_id is new",), ("reset complete",), ("IMPLEMENTATION_EXCEPTION",)),
    AttachState.STAND_SETTLE: StateMetadata("settle with fixed lower body command", ("stand_settle",), _COMMON_FORBIDDEN, _COMMON_OBS, ("RESET complete",), ("settled",), ("ROBOT_FALL", "ROBOT_BAD_TILT", "NONFINITE")),
    AttachState.PRECONTACT: StateMetadata("hold a non-penetrating precontact pose", ("hold",), _COMMON_FORBIDDEN, _COMMON_OBS, ("stand settled",), ("precontact verified",), ("PRECONTACT_COLLISION", "NONFINITE")),
    AttachState.APPROACH_NORMAL: StateMetadata("approach along the rear normal only", ("normal_approach",), _COMMON_FORBIDDEN, _COMMON_OBS, ("precontact verified",), ("contact verification entered",), ("PRECONTACT_COLLISION", "ROBOT_FALL", "NONFINITE")),
    AttachState.BILATERAL_CONTACT_VERIFY: StateMetadata("verify both palms and stable object", ("hold",), _COMMON_FORBIDDEN, _COMMON_OBS + ("object_linear_speed_mps", "object_yaw_rate_radps"), ("normal approach complete",), ("bilateral contact verified",), ("LEFT_CONTACT_MISSING", "RIGHT_CONTACT_MISSING", "BILATERAL_CONTACT_TIMEOUT", "OBJECT_TRANSLATION_DURING_ATTACH", "OBJECT_YAW_DURING_ATTACH")),
    AttachState.ATTACHED_HOLD: StateMetadata("hold bilateral contact without pushing", ("hold",), _COMMON_FORBIDDEN, _COMMON_OBS, ("attach gate accepted",), ("hold complete",), ("LEFT_CONTACT_LOST", "RIGHT_CONTACT_LOST", "FORBIDDEN_BODY_BOX_COLLISION", "ROBOT_FALL", "ROBOT_BAD_TILT", "NONFINITE")),
    AttachState.PASS: StateMetadata("terminal valid attachment", (), ("push", "reset"), _COMMON_OBS, ("hold complete",), (), ()),
    AttachState.FAIL: StateMetadata("terminal physical or evidence failure", (), ("push", "reset"), _COMMON_OBS, ("failure reason recorded",), (), ()),
}


def next_state(
    current: AttachState,
    requested: AttachState,
    *,
    failure_reason: str | None = None,
) -> AttachState:
    if requested not in LEGAL_TRANSITIONS[current]:
        raise ValueError(f"illegal attach transition: {current.value} -> {requested.value}")
    if requested is AttachState.FAIL:
        valid_reasons = PHYSICAL_FAILURE_REASONS | INVALID_REASONS
        if failure_reason not in valid_reasons:
            raise ValueError("FAIL transition requires one exact registered failure reason")
    elif failure_reason is not None:
        raise ValueError("failure_reason is only valid for a FAIL transition")
    return requested
