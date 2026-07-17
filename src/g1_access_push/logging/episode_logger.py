# src/g1_access_push/logging/episode_logger.py
"""Episode 日志与自动回放。对应文档 8.9 日志要求、式(10.1) 版本元组。
每个 episode 必须可重放,才能区分规划失败/状态估计失败/局部执行失败/WBC 失败。
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class EpisodeLog:
    seed: int
    geometry: Dict[str, Any]          # 箱子/门/障碍几何
    physics: Dict[str, Any]           # 质量、摩擦、质心
    wbc_id: str                       # WBCID
    planner_version: str
    template_sequence: List[str] = field(default_factory=list)
    edge_checks: List[Dict[str, Any]] = field(default_factory=list)  # 每条边检查结果与耗时
    states: List[Dict[str, Any]] = field(default_factory=list)       # 状态估计/命令/接触/碰撞
    failure_code: int = 0
    video_timestamp: float = 0.0
    created_at: float = field(default_factory=time.time)

    def add_state(self, **kw):
        self.states.append(kw)

    def add_edge_check(self, **kw):
        self.edge_checks.append(kw)

    def save(self, out_dir: str = "experiment_logs") -> str:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fname = Path(out_dir) / f"ep_seed{self.seed}_{int(self.created_at)}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        return str(fname)

    @staticmethod
    def load(path: str) -> "EpisodeLog":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EpisodeLog(**data)