# src/g1_access_push/task/success_failure.py
"""任务成功/失败定义。对应文档式(3.7) 与 F1-F8 失败分类。"""
from __future__ import annotations
import numpy as np
from enum import Enum
from dataclasses import dataclass
from g1_access_push.task.coordinate_frames import wrap
from g1_access_push.geometry.doorway import Door, object_clear_door


class FailureCode(Enum):
    """3.5 节:互斥、有优先级的失败分类。"""
    NONE = 0
    F1_NO_PLAN = 1          # 无规划解
    F2_UNREACHABLE = 2      # 不可到达站位
    F3_IK_BODY = 3          # 手臂连续 IK / 身体碰撞失败
    F4_CONTACT_LOST = 4     # 接触丢失或箱子响应方向错误
    F5_FALL_TORQUE = 5      # 跌倒或力矩超限
    F6_HIT_DOORFRAME = 6    # 箱子/机器人碰门框
    F7_STUCK_IN_DOOR = 7    # 门内卡住且无法非拉式恢复
    F8_TIMEOUT = 8          # 超时


@dataclass
class TaskState:
    q_o: tuple            # (xo, yo, theta_o) 箱子位姿
    robot_clear_door: bool  # 机器人包络最后一点是否越过门后
    fall: bool = False
    collision: bool = False


@dataclass
class TaskGoal:
    p_g: tuple            # (xg, yg) 目标位置
    theta_g: float        # 目标朝向
    eps_p: float = 0.05   # 位置容差
    eps_theta: float = 0.087  # 约 5 度


def check_success(state: TaskState, goal: TaskGoal, door: Door,
                  L: float, W: float) -> bool:
    """式(3.7):
    Success = ObjectClearDoor & RobotClearDoor & |po-pg|<=eps_p
              & |wrap(theta_o-theta_g)|<=eps_theta & !Fall & !Collision.
    """
    po = np.array(state.q_o[:2])
    pos_ok = np.linalg.norm(po - np.array(goal.p_g)) <= goal.eps_p
    yaw_ok = abs(wrap(state.q_o[2] - goal.theta_g)) <= goal.eps_theta
    obj_clear = object_clear_door(door, state.q_o, L, W)
    return bool(obj_clear and state.robot_clear_door and pos_ok
                and yaw_ok and not state.fall and not state.collision)


def classify_failure(*, plan_found: bool, station_reachable: bool,
                     ik_ok: bool, contact_ok: bool, fell: bool,
                     hit_doorframe: bool, stuck_in_door: bool,
                     timed_out: bool) -> FailureCode:
    """按固定优先级分类失败(3.5 节:失败必须互斥分类)。"""
    if timed_out:
        return FailureCode.F8_TIMEOUT
    if not plan_found:
        return FailureCode.F1_NO_PLAN
    if not station_reachable:
        return FailureCode.F2_UNREACHABLE
    if not ik_ok:
        return FailureCode.F3_IK_BODY
    if not contact_ok:
        return FailureCode.F4_CONTACT_LOST
    if fell:
        return FailureCode.F5_FALL_TORQUE
    if hit_doorframe:
        return FailureCode.F6_HIT_DOORFRAME
    if stuck_in_door:
        return FailureCode.F7_STUCK_IN_DOOR
    return FailureCode.NONE