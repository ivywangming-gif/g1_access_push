# src/g1_access_push/task/coordinate_frames.py
"""SE(2) 坐标系与变换库。对应文档 3.1、式(3.1)-(3.2)。
坐标系:{W}世界, {D}门, {O}箱子, {B}base, {HL}/{HR}左右手。
单位:米、弧度。角度与长度不直接相加(见附录 C.2)。
"""
from __future__ import annotations
import numpy as np


def wrap(angle: float) -> float:
    """把角度归一化到 (-pi, pi]。"""
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def rot2d(theta: float) -> np.ndarray:
    """式(3.2) 的 2x2 旋转矩阵 R(theta)。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def se2_matrix(x: float, y: float, theta: float) -> np.ndarray:
    """式(3.1):由 (x, y, theta) 构造 3x3 齐次变换 T。"""
    T = np.eye(3)
    T[:2, :2] = rot2d(theta)
    T[:2, 2] = [x, y]
    return T


def se2_inverse(T: np.ndarray) -> np.ndarray:
    """SE(2) 逆变换。"""
    R = T[:2, :2]
    p = T[:2, 2]
    Ti = np.eye(3)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -R.T @ p
    return Ti


def se2_compose(*Ts: np.ndarray) -> np.ndarray:
    """依次左乘:se2_compose(A, B, C) = A @ B @ C。"""
    out = np.eye(3)
    for T in Ts:
        out = out @ T
    return out


def se2_from_pose(pose) -> np.ndarray:
    """(x, y, theta) -> T。"""
    x, y, theta = pose
    return se2_matrix(x, y, theta)


def pose_from_se2(T: np.ndarray):
    """T -> (x, y, theta)。"""
    x, y = T[0, 2], T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return float(x), float(y), float(theta)


def transform_point(T: np.ndarray, p) -> np.ndarray:
    """用 T 变换一个 2D 点。"""
    ph = np.array([p[0], p[1], 1.0])
    return (T @ ph)[:2]


def hand_world_pose(q_o, T_OC: np.ndarray, T_CH: np.ndarray) -> np.ndarray:
    """测试#1 的变换链:T_WH = T_WO(q_o) @ T_OC @ T_CH。
    q_o=(xo,yo,theta_o) 箱子位姿;T_OC 接触点相对箱子;T_CH 手掌相对接触。
    对应式(6.16) 的离散化形式。
    """
    T_WO = se2_from_pose(q_o)
    return se2_compose(T_WO, T_OC, T_CH)