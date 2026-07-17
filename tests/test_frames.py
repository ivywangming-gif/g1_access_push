# tests/test_frames.py
import numpy as np
from g1_access_push.task.coordinate_frames import (
    wrap, se2_matrix, se2_inverse, se2_compose, hand_world_pose, pose_from_se2)


def test_inverse_and_compose():
    T = se2_matrix(1.0, 2.0, 0.7)
    I = se2_compose(T, se2_inverse(T))
    assert np.allclose(I, np.eye(3), atol=1e-12)


def test_wrap():
    assert np.isclose(wrap(3 * np.pi), np.pi) or np.isclose(wrap(3 * np.pi), -np.pi)
    assert np.isclose(wrap(0.0), 0.0)


def test_two_toc_tch_chain():
    # 测试#1:TWO -> TOC -> TCH 链与手工计算一致
    q_o = (1.0, 2.0, np.pi / 2)
    T_OC = se2_matrix(-0.4, 0.1, 0.0)   # 接触点相对箱子
    T_CH = se2_matrix(0.05, 0.0, 0.0)   # 手掌相对接触
    T_WH = hand_world_pose(q_o, T_OC, T_CH)
    x, y, th = pose_from_se2(T_WH)
    # 手工推导:R(pi/2)@(-0.4,0.1)=(-0.1,-0.4)+(1,2)=(0.9,1.6);
    # 再加 R(pi/2)@(0.05,0)=(0,0.05) -> (0.9,1.65),朝向 pi/2
    assert np.isclose(x, 0.9, atol=1e-9)
    assert np.isclose(y, 1.65, atol=1e-9)
    assert np.isclose(wrap(th - np.pi / 2), 0.0, atol=1e-9)