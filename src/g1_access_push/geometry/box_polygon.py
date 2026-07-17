# src/g1_access_push/geometry/box_polygon.py
"""矩形箱子的平面几何与门宽投影。对应文档 4.1、式(3.5)、(4.2)-(4.5)。"""
from __future__ import annotations
import numpy as np
from g1_access_push.task.coordinate_frames import wrap, se2_from_pose, transform_point


def box_vertices_object(L: float, W: float) -> np.ndarray:
    """式(4.2):箱子四个顶点在箱子坐标系 {O} 中的坐标。
    xO 沿长边 L,yO 沿短边 W。
    """
    hx, hy = L / 2.0, W / 2.0
    return np.array([[+hx, +hy], [+hx, -hy], [-hx, -hy], [-hx, +hy]], dtype=float)


def box_vertices_world(q_o, L: float, W: float) -> np.ndarray:
    """把四个顶点变换到世界系。"""
    T_WO = se2_from_pose(q_o)
    return np.array([transform_point(T_WO, v) for v in box_vertices_object(L, W)])


def doorwidth_projection(L: float, W: float, alpha: float) -> float:
    """式(4.5):w_perp(alpha) = L|sin a| + W|cos a|。
    alpha = wrap(theta_o - theta_D)。
    alpha=0 -> W(箱子长边顺着穿门方向);alpha=pi/2 -> L。
    """
    a = wrap(alpha)
    return L * abs(np.sin(a)) + W * abs(np.cos(a))


def half_projection_along_door(L: float, W: float, alpha: float) -> float:
    """式(3.5):h_par(alpha) = 0.5 (L|cos a| + W|sin a|)。
    箱子沿穿门方向的半投影长度,用于判断箱尾是否越过出口平面。
    """
    a = wrap(alpha)
    return 0.5 * (L * abs(np.cos(a)) + W * abs(np.sin(a)))


def _cross2(r: np.ndarray, f: np.ndarray) -> float:
    """2D 叉积 r_x*f_y - r_y*f_x。"""
    return float(r[0] * f[1] - r[1] * f[0])


def planar_wrench(contacts: np.ndarray, forces: np.ndarray):
    """测试#3:双手接触在箱子坐标系产生的平面 wrench (Fx, Fy, tau)。
    对应第4章双手 wrench 与式(6.21) 的物体 twist 方向粗筛。

    约定(必须与仿真核对):
      - 箱子坐标系 xO 沿长边、yO 沿短边;
      - 后表面居中模板:两手在 xO=-L/2 面上,沿 +xO 推;
      - 左手在 +yO 侧,右手在 -yO 侧。
    结论:左手比右手推得更用力 -> tau < 0(箱子 +xO 朝 -yO 偏转,顺时针)。
    若 Debug-G1 仿真里符号相反,只需在这里翻转叉积号并同步改注释。
    """
    contacts = np.asarray(contacts, dtype=float)
    forces = np.asarray(forces, dtype=float)
    F = forces.sum(axis=0)
    tau = sum(_cross2(contacts[i], forces[i]) for i in range(len(contacts)))
    return float(F[0]), float(F[1]), float(tau)