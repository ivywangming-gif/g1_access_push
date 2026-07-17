"""Crash-tolerant JSONL episode logging for deterministic failure analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported log value type: {type(value).__name__}")


class EpisodeLogger:
    """Write one self-contained episode directory.

    Files:
    - ``metadata.json``: version tuple, seed, geometry and initial conditions.
    - ``resolved_config.json``: immutable resolved settings used by the run.
    - ``steps.jsonl``: timestamped state/command/event records.
    - ``summary.json``: terminal outcome and failure code.
    """

    def __init__(
        self,
        root_dir: str | Path,
        episode_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        if not episode_id or "/" in episode_id:
            raise ValueError("episode_id must be non-empty and may not contain '/'")
        self.episode_dir = Path(root_dir).expanduser().resolve() / episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self._steps_path = self.episode_dir / "steps.jsonl"
        self._finalized = False

        metadata_payload = {
            "schema_version": 1,
            "episode_id": episode_id,
            "started_at_utc": datetime.now(UTC).isoformat(),
            **dict(metadata),
        }
        self._write_json(self.episode_dir / "metadata.json", metadata_payload)

    def write_resolved_config(self, config: Mapping[str, Any]) -> None:
        self._ensure_open()
        self._write_json(self.episode_dir / "resolved_config.json", dict(config))

    def append_step(self, timestamp_s: float, record: Mapping[str, Any]) -> None:
        self._ensure_open()
        if timestamp_s < 0.0 or not np.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite and non-negative")
        payload = {"timestamp_s": float(timestamp_s), **dict(record)}
        with self._steps_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
            stream.flush()

    def finalize(self, summary: Mapping[str, Any]) -> Path:
        self._ensure_open()
        payload = {
            "finished_at_utc": datetime.now(UTC).isoformat(),
            **dict(summary),
        }
        self._write_json(self.episode_dir / "summary.json", payload)
        self._finalized = True
        return self.episode_dir

    @staticmethod
    def load_steps(episode_dir: str | Path) -> list[dict[str, Any]]:
        path = Path(episode_dir) / "steps.jsonl"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    @staticmethod
    def load_json(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("EpisodeLogger has already been finalized")

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
