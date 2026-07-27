"""Static contracts for S2-00 attach-only development.

The module is deliberately independent of Isaac, Omniverse, and controller code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureReason(str, Enum):
    PRECONTACT_COLLISION = "PRECONTACT_COLLISION"
    LEFT_CONTACT_MISSING = "LEFT_CONTACT_MISSING"
    RIGHT_CONTACT_MISSING = "RIGHT_CONTACT_MISSING"
    BILATERAL_CONTACT_TIMEOUT = "BILATERAL_CONTACT_TIMEOUT"
    EXCESSIVE_CONTACT_IMPULSE = "EXCESSIVE_CONTACT_IMPULSE"
    OBJECT_TRANSLATION_DURING_ATTACH = "OBJECT_TRANSLATION_DURING_ATTACH"
    OBJECT_YAW_DURING_ATTACH = "OBJECT_YAW_DURING_ATTACH"
    LEFT_CONTACT_LOST = "LEFT_CONTACT_LOST"
    RIGHT_CONTACT_LOST = "RIGHT_CONTACT_LOST"
    FORBIDDEN_BODY_BOX_COLLISION = "FORBIDDEN_BODY_BOX_COLLISION"
    ROBOT_FALL = "ROBOT_FALL"
    ROBOT_BAD_TILT = "ROBOT_BAD_TILT"
    JOINT_LIMIT_VIOLATION = "JOINT_LIMIT_VIOLATION"
    TORQUE_LIMIT_VIOLATION = "TORQUE_LIMIT_VIOLATION"
    NONFINITE = "NONFINITE"


class InvalidReason(str, Enum):
    IMPLEMENTATION_EXCEPTION = "IMPLEMENTATION_EXCEPTION"
    MISSING_RESULT_JSON = "MISSING_RESULT_JSON"
    MISSING_TRACE = "MISSING_TRACE"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    TERMINATION_API_AMBIGUOUS = "TERMINATION_API_AMBIGUOUS"
    ACTION_CONTRACT_UNCERTIFIED = "ACTION_CONTRACT_UNCERTIFIED"
    EVIDENCE_INCOMPLETE_BEFORE_TIMEOUT = "EVIDENCE_INCOMPLETE_BEFORE_TIMEOUT"
    MULTIPLE_ISAAC_PROCESSES = "MULTIPLE_ISAAC_PROCESSES"
    CONFIG_UNRESOLVED = "CONFIG_UNRESOLVED"


class ResultStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"
    NOT_RUN = "NOT_RUN"


PHYSICAL_FAILURE_REASONS = frozenset(item.value for item in FailureReason)
INVALID_REASONS = frozenset(item.value for item in InvalidReason)


def _is_unresolved(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("status") == "UNRESOLVED"


def unresolved_parameters(config: Mapping[str, Any]) -> list[str]:
    """Return stable dotted paths for every recursively unresolved parameter."""

    found: list[str] = []

    def walk(value: Any, path: str) -> None:
        if _is_unresolved(value):
            found.append(path)
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(config, "")
    return sorted(found)


@dataclass(frozen=True)
class AttachGateEvidence:
    left_contact: bool
    right_contact: bool
    object_linear_speed_mps: float
    object_yaw_rate_radps: float
    robot_stable: bool
    no_forbidden_collision: bool


def _resolved_threshold(thresholds: Mapping[str, Any], key: str) -> float | None:
    raw = thresholds.get(key, "UNRESOLVED")
    if isinstance(raw, Mapping):
        if raw.get("status") == "UNRESOLVED":
            return None
        raw = raw.get("value", "UNRESOLVED")
    if raw in ("UNRESOLVED", None, ""):
        return None
    return float(raw)


def evaluate_attach_gate(
    evidence: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate only the data contract; unresolved thresholds never yield PASS."""

    required = (
        "left_contact",
        "right_contact",
        "object_linear_speed_mps",
        "object_yaw_rate_radps",
        "robot_stable",
        "no_forbidden_collision",
    )
    missing = [key for key in required if key not in evidence]
    if missing:
        return {"status": "INVALID_EVIDENCE", "missing_fields": missing}
    linear_limit = _resolved_threshold(thresholds, "object_linear_speed_threshold")
    yaw_limit = _resolved_threshold(thresholds, "object_yaw_rate_threshold")
    unresolved = [
        key
        for key, value in (
            ("object_linear_speed_threshold", linear_limit),
            ("object_yaw_rate_threshold", yaw_limit),
        )
        if value is None
    ]
    if unresolved:
        return {"status": "CONTRACT_UNRESOLVED", "unresolved_thresholds": unresolved}
    accepted = bool(
        evidence["left_contact"]
        and evidence["right_contact"]
        and float(evidence["object_linear_speed_mps"]) <= linear_limit
        and abs(float(evidence["object_yaw_rate_radps"])) <= yaw_limit
        and evidence["robot_stable"]
        and evidence["no_forbidden_collision"]
    )
    return {"status": "PASS" if accepted else "FAIL", "accepted": accepted}
