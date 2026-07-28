"""Pure process/config helpers for S2-01; safe without Isaac/Kit."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_controller_checkpoint(
    checkpoint_path: str | Path,
    actual_sha256: str | None,
    expected_sha256: str,
    forbidden_sha256s: list[str] | tuple[str, ...],
    forbidden_basenames: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Validate only the resolved checkpoint identity, never unrelated config text."""
    path = Path(checkpoint_path).expanduser().resolve()
    exists = path.is_file()
    actual = actual_sha256 if actual_sha256 is not None else (sha256_path(path) if exists else None)
    forbidden_sha_match = actual in set(forbidden_sha256s) if actual else False
    forbidden_path_match = path.name.lower() in {name.lower() for name in forbidden_basenames}
    sha_match = actual == expected_sha256
    if not exists:
        reason = "CHECKPOINT_NOT_FOUND"
    elif forbidden_sha_match or forbidden_path_match:
        reason = "FORBIDDEN_CHECKPOINT_SELECTED"
    elif not sha_match:
        reason = "CHECKPOINT_SHA_MISMATCH"
    else:
        reason = None
    return {
        "status": "PASS" if reason is None else "FAIL",
        "reason": reason,
        "controller_checkpoint_path": str(path),
        "controller_checkpoint_actual_sha256": actual,
        "controller_checkpoint_expected_sha256": expected_sha256,
        "controller_checkpoint_sha_match": sha_match,
        "controller_checkpoint_forbidden_sha_match": forbidden_sha_match,
        "controller_checkpoint_forbidden_path_match": forbidden_path_match,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _trace_frame_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def derive_effective_runner_status(run_root: Path, raw_rc: int, expected_frames: int) -> dict[str, Any]:
    """Derive authoritative status without importing Isaac or trusting Kit's raw RC."""
    marker_present = (run_root / "implementation_exception.json").is_file()
    runner = _read_json(run_root / "runner_status.json")
    trace_frames = _trace_frame_count(run_root / "trace.jsonl")
    if marker_present:
        reason = "IMPLEMENTATION_EXCEPTION"
    elif runner is None:
        reason = "MISSING_RUNNER_STATUS"
    elif runner.get("status") != "COMPLETE":
        reason = str(runner.get("primary_reason") or "RUNNER_NOT_COMPLETE")
    elif runner.get("environment_created") is not True:
        reason = "ENVIRONMENT_NOT_CREATED"
    elif trace_frames is None:
        reason = "MISSING_TRACE"
    elif int(runner.get("observed_frames", -1)) != expected_frames or trace_frames != expected_frames:
        reason = "INCOMPLETE_TRACE"
    elif raw_rc != 0:
        reason = "RUNNER_RAW_RC_NONZERO"
    else:
        reason = None
    effective_rc = 0 if reason is None else (raw_rc if reason == "RUNNER_RAW_RC_NONZERO" else 1)
    return {
        "schema_version": 1,
        "runner_raw_rc": raw_rc,
        "runner_effective_rc": effective_rc,
        "status": "COMPLETE" if reason is None else "INVALID",
        "primary_reason": reason,
        "expected_frames": expected_frames,
        "observed_frames": None if runner is None else runner.get("observed_frames"),
        "trace_frames": trace_frames,
        "environment_created": None if runner is None else runner.get("environment_created"),
        "implementation_exception_present": marker_present,
    }


def persist_effective_runner_status(run_root: Path, raw_rc: int, expected_frames: int) -> dict[str, Any]:
    result = derive_effective_runner_status(run_root, raw_rc, expected_frames)
    process_rc = run_root / "process_rc"
    process_rc.mkdir(parents=True, exist_ok=True)
    (process_rc / "runner_raw.txt").write_text(f"{raw_rc}\n", encoding="utf-8")
    (process_rc / "runner_effective.txt").write_text(f"{result['runner_effective_rc']}\n", encoding="utf-8")
    (process_rc / "runner.txt").write_text(f"{result['runner_effective_rc']}\n", encoding="utf-8")
    temporary = run_root / "runner_effective_status.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(run_root / "runner_effective_status.json")
    return result


def build_scene_config_instance(base_scene_cfg: Any, *, terrain_cfg: Any, additions: dict[str, Any]) -> Any:
    """Copy one resolved scene instance and add S2-01 entities without class-field access."""
    if isinstance(base_scene_cfg, type):
        raise TypeError("SCENE_CONFIG_INSTANCE_REQUIRED")
    if not hasattr(base_scene_cfg, "terrain") or not hasattr(base_scene_cfg, "robot"):
        raise TypeError("SCENE_CONFIG_INSTANCE_REQUIRED")
    result = base_scene_cfg.copy() if callable(getattr(base_scene_cfg, "copy", None)) else copy.deepcopy(base_scene_cfg)
    if result is base_scene_cfg:
        result = copy.deepcopy(base_scene_cfg)
    result.terrain = copy.deepcopy(terrain_cfg)
    for name, value in additions.items():
        setattr(result, name, copy.deepcopy(value))
    return result


def write_implementation_exception(path: Path, exc: BaseException, context: dict[str, Any] | None = None) -> None:
    payload = {
        "schema_version": 1,
        "status": "INVALID",
        "primary_reason": "IMPLEMENTATION_EXCEPTION",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        **(context or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def guarded_process_main(
    main_fn: Callable[[], Any],
    cleanup_fn: Callable[[], Any],
    *,
    exception_path: Path,
    context_fn: Callable[[], dict[str, Any]] | None = None,
) -> int:
    """Return nonzero for main or cleanup exceptions; cleanup never masks a prior exception."""
    failure: BaseException | None = None
    rc = 0
    try:
        value = main_fn()
        if isinstance(value, int):
            rc = value
    except BaseException as exc:
        failure = exc
        rc = int(exc.code) if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        rc = rc or 1
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    try:
        cleanup_fn()
    except BaseException as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        if failure is None:
            failure = exc
            rc = 1
    if failure is not None:
        write_implementation_exception(exception_path, failure, context_fn() if context_fn else None)
    return rc
