# tests/test_success_failure.py
import numpy as np
from g1_access_push.geometry.doorway import Door
from g1_access_push.task.success_failure import (
    TaskState, TaskGoal, check_success, classify_failure, FailureCode)


def make_door():
    # 门在原点,穿门方向 = +x,门宽 0.9,出口平面 x_exit=0
    return Door(xD=0.0, yD=0.0, theta_D=0.0, D=0.9, x_exit=0.0)


def test_success_requires_object_and_robot_clear():
    # 测试#4:success 必须箱子和机器人都完全清空门
    door = make_door()
    L, W = 1.0, 0.4
    goal = TaskGoal(p_g=(1.5, 0.0), theta_g=0.0)

    # 箱子中心在门后 1.5m、朝向顺门,半投影 h_par=L/2=0.5 -> 箱尾在 1.0 > 0,清空
    q_o = (1.5, 0.0, 0.0)

    # 机器人未清空 -> 失败
    s_robot_not = TaskState(q_o=q_o, robot_clear_door=False)
    assert check_success(s_robot_not, goal, door, L, W) is False

    # 机器人也清空 -> 成功
    s_ok = TaskState(q_o=q_o, robot_clear_door=True)
    assert check_success(s_ok, goal, door, L, W) is True

    # 箱子还没越过出口平面(中心太靠前) -> 失败
    s_obj_not = TaskState(q_o=(0.3, 0.0, 0.0), robot_clear_door=True)
    goal_near = TaskGoal(p_g=(0.3, 0.0), theta_g=0.0)
    assert check_success(s_obj_not, goal_near, door, L, W) is False


def test_failure_classification_distinct():
    # 测试#5:失败日志能区分 无规划解 / 碰撞 / 跌倒 / 接触丢失
    base = dict(plan_found=True, station_reachable=True, ik_ok=True,
                contact_ok=True, fell=False, hit_doorframe=False,
                stuck_in_door=False, timed_out=False)

    assert classify_failure(**{**base, "plan_found": False}) == FailureCode.F1_NO_PLAN
    assert classify_failure(**{**base, "contact_ok": False}) == FailureCode.F4_CONTACT_LOST
    assert classify_failure(**{**base, "fell": True}) == FailureCode.F5_FALL_TORQUE
    assert classify_failure(**{**base, "hit_doorframe": True}) == FailureCode.F6_HIT_DOORFRAME
    # 四类互不相同
    codes = {FailureCode.F1_NO_PLAN, FailureCode.F4_CONTACT_LOST,
             FailureCode.F5_FALL_TORQUE, FailureCode.F6_HIT_DOORFRAME}
    assert len(codes) == 4