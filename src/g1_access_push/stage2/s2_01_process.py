"""Pure process/config helpers for S2-01; safe without Isaac/Kit."""

from __future__ import annotations

import copy
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


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
    except BaseException as exc:  # process boundary deliberately includes SystemExit
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
