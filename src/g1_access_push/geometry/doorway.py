# src/g1_access_push/geometry/doorway.py
"""门几何:门坐标系 {D}、门宽净空、箱子清空门判定。
对应文档 3.1、式(3.6)、(4.6)-(4.7)、(8.4)。
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from g1_access_push.task.coordinate_frames import wrap, se2_from_pose, se2_inverse, transform_point


@dataclass
class Door:
    """门:中心 (xD, yD),穿门方向角 theta_D,门宽 D,出口平面 x_exit(门坐标系 xD)。"""
    xD: float
    yD: float
    theta_D: float
    D: float          # 门宽
    x_exit: float     # 门坐标系下出口平面位置
    thickness: float = 0.1

    def T_WD(self) -> np.ndarray:
        return se2_from_pose((self.xD, self.yD, self.theta_D))

    def object_in_door_frame(self, q_o):
        """把箱子位姿 (xo,yo,theta_o) 表达到门坐标系 {D}。返回 (xoD, yoD, alpha)。"""
        T_DW = se2_inverse(self.T_WD())
        p_D = transform_point(T_DW, (q_o[0], q_o[1]))
        alpha = wrap(q_o[2] - self.theta_D)   # 式(4.1)
        return float(p_D[0]), float(p_D[1]), float(alpha)


def lateral_clearance(door: Door, q_o, L: float, W: float, delta: float = 0.0) -> float:
    """式(8.4):c_D = D/2 - |yoD| - 0.5 w_perp(alpha) - delta。
    c_D < 0 表示即使只看当前投影也已不可行。
    """
    from g1_access_push.geometry.box_polygon import doorwidth_projection
    _, yoD, alpha = door.object_in_door_frame(q_o)
    w_perp = doorwidth_projection(L, W, alpha)
    return door.D / 2.0 - abs(yoD) - 0.5 * w_perp - delta


def object_clear_door(door: Door, q_o, L: float, W: float, delta_exit: float = 0.0) -> bool:
    """式(3.6):xoD - h_par(alpha) >= x_exit + delta_exit。
    即箱子最后端也越过出口平面。
    """
    from g1_access_push.geometry.box_polygon import half_projection_along_door
    xoD, _, alpha = door.object_in_door_frame(q_o)
    h_par = half_projection_along_door(L, W, alpha)
    return (xoD - h_par) >= (door.x_exit + delta_exit)