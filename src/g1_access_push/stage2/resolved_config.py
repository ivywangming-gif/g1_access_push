"""Deterministic YAML-to-JSON resolver for the S2-00 contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .contract import unresolved_parameters


FORBIDDEN_TRUE = ("isaac_execution_enabled", "planner_enabled", "pushing_enabled")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_parameter(path: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict) or "status" not in value or "value" not in value:
        return
    status = value["status"]
    if status == "UNRESOLVED" and value["value"] not in ("UNRESOLVED",):
        errors.append(f"{path}: unresolved value must be the UNRESOLVED sentinel")
    if status == "FROZEN":
        for key in ("unit", "source"):
            if not value.get(key): errors.append(f"{path}: frozen parameter lacks {key}")


def _walk(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        _validate_parameter(path, value, errors)
        for key, child in value.items():
            _walk(child, f"{path}.{key}" if path else str(key), errors)
    elif isinstance(value, list):
        for index, child in enumerate(value): _walk(child, f"{path}[{index}]", errors)


def resolve_config(config_path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    _walk(config, "", errors)
    if config.get("geometry", {}).get("door_width", {}).get("value") is not None and not config.get("geometry", {}).get("door_disabled_in_stage2", {}).get("value"):
        errors.append("door_width requires door_disabled_in_stage2=true")
    for key in FORBIDDEN_TRUE:
        if config.get(key) is True: errors.append(f"{key} must be false")
    if config.get("geometry", {}).get("doorway_enabled", {}).get("value") is True:
        errors.append("doorway_enabled must be false")
    com = config.get("object_physics", {}).get("box_com_height", {})
    if com.get("value") == 0.60 and com.get("source") == "UNRESOLVED":
        errors.append("geometric-center CoM cannot be silently accepted")
    if errors: raise ValueError("invalid S2-00 config: " + "; ".join(errors))
    unresolved = unresolved_parameters(config)
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    digest_payload = {"config": config, "repository_head": head, "branch": branch}
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = {
        "schema_version": config["schema_version"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_config_path": str(path),
        "source_config_sha256": _sha256(path),
        "repository_head": head,
        "branch": branch,
        "stage": config["stage"],
        "runnable": False,
        "frozen_parameters": config,
        "unresolved_parameters": unresolved,
        "unresolved_count": len(unresolved),
        "contradictions": [],
        "unit_audit": "PASSED_PARAMETER_METADATA_SCAN",
        "source_audit": "PASSED_PARAMETER_METADATA_SCAN",
        "contract_digest_sha256": digest,
        "qualification_state": "CONTRACT_ONLY_NOT_EXECUTED",
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
