# tests/test_door_projection.py
import numpy as np
from g1_access_push.geometry.box_polygon import (
    doorwidth_projection, half_projection_along_door, planar_wrench)


def test_projection_rotate_90_deg():
    # 测试#2:箱子旋转 90 度后,门宽投影从 W 变为 L
    L, W = 1.2, 0.5
    assert np.isclose(doorwidth_projection(L, W, 0.0), W)          # 顺门:投影 = W
    assert np.isclose(doorwidth_projection(L, W, np.pi / 2), L)    # 转 90 度:投影 = L
    assert np.isclose(half_projection_along_door(L, W, 0.0), L / 2)


def test_differential_yaw_sign():
    # 测试#3:左右手差动力的 yaw 符号(约定见 box_polygon.planar_wrench)
    L, W, d = 1.2, 0.4, 0.15
    contacts = np.array([[-L / 2, +d],   # 左手 (+yO)
                         [-L / 2, -d]])  # 右手 (-yO)
    # 左手推得更用力
    forces = np.array([[6.0, 0.0], [4.0, 0.0]])
    Fx, Fy, tau = planar_wrench(contacts, forces)
    assert Fx > 0 and np.isclose(Fy, 0.0)
    assert tau < 0.0    # 左强 -> 顺时针 -> 负 yaw

    # 右手推得更用力 -> 符号翻转
    _, _, tau2 = planar_wrench(contacts, np.array([[4.0, 0.0], [6.0, 0.0]]))
    assert tau2 > 0.0

    # 对称推动 -> 无净力矩
    _, _, tau3 = planar_wrench(contacts, np.array([[5.0, 0.0], [5.0, 0.0]]))
    assert np.isclose(tau3, 0.0)