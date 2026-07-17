# src/g1_access_push/control/wbc_adapter.py
"""规划器 <-> WBC 的输入输出接口冻结。
对应文档式(3.4)、式(10.3) planner->executor、式(10.4) executor->planner。
Stage 0 只定义数据契约,不做真实控制。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, List
import numpy as np


class ExecStatus(Enum):
    """式(10.4):status 只能来自预定义枚举,禁止自由文本解析。"""
    RUNNING = "running"
    SUCCESS = "success"
    STOPPED_SAFE = "stopped_safe"
    FAILED = "failed"


@dataclass
class EdgeCommand:
    """式(10.3) Me = (q_nom_o(t), lambda, a, Te, RB_e(t), r_geom, stopRules)。
    planner -> executor 的一条边执行消息。
    """
    q_nom_o: Callable[[float], np.ndarray]   # 名义箱子轨迹 t->(x,y,theta)
    template_id: str                         # lambda:交互模板
    action: str                              # a:局部动作 (push/turn/stop/reposition)
    Te: float                                # 边持续时间
    base_corridor: Callable[[float], np.ndarray]  # RB_e(t):base 走廊参考
    r_geom: float                            # 几何执行余量
    stop_rules: dict = field(default_factory=dict)  # 硬/软阈值


@dataclass
class ExecFeedback:
    """式(10.4) Ye = (status, q_o, q_b, chiL, chiR, zeta, failureCode, t_elapsed)。
    executor -> planner 的执行反馈。
    """
    status: ExecStatus
    q_o: np.ndarray            # 箱子实测位姿
    q_b: np.ndarray            # base 实测位姿
    chi_L: float               # 左手接触状态 [0,1]
    chi_R: float               # 右手接触状态 [0,1]
    zeta: float                # 式(8.8) 执行风险指标
    failure_code: int          # FailureCode.value
    t_elapsed: float


class WBCAdapter:
    """Stage 0 占位适配器:只校验接口形状,不驱动仿真。
    后续阶段把 send_edge 接到冻结的 AGILE G1 WBC。
    """
    def __init__(self, wbc_id: str):
        self.wbc_id = wbc_id   # WBCID:checkpoint+频率+观测+增益版本(见式7.2)

    def send_edge(self, cmd: EdgeCommand) -> None:
        assert callable(cmd.q_nom_o)
        assert cmd.Te > 0
        # Stage 0:不执行,仅冻结契约

    def make_feedback(self, **kw) -> ExecFeedback:
        return ExecFeedback(**kw)