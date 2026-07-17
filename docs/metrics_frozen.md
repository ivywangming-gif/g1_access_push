# 指标冻结(Stage 0,只读,后续阶段不得随意改动)

来源:研究规划文档 §11.5。任何修改都必须记录在 weekly_log 并升级版本号。

## 11.5.1 任务指标
- episode 成功率(式 3.7)
- 规划成功但执行失败率
- 完成时间 / 物体路径长度
- 最终位置误差 |po - pg|、yaw 误差 |wrap(θo - θg)|
- reposition 次数
- 门内停止 / 碰撞 / 卡住率

## 11.5.2 规划正确性指标
- 不可执行计划率 R_inexec (式 11.1)
- 站位瞬移错误率 R_teleport (式 11.2) —— 本文方法必须为 0

## 11.5.3 计算指标
- 首解 / 最终验证 / 重规划时间的 median/p90/p95
- 每次规划的昂贵 IK/WBC rollout 数、cache hit rate
- 相同 CPU/GPU 与线程下的 wall-clock

## 11.5.4 执行与安全指标
- fall rate;箱-门/机器人-门/机器人-箱非预期碰撞率
- minimum clearance;双手接触保持率与最长连续丢失时间
- 峰值接触力、物体加速度、joint torque ratio
- object tracking integrated error E_o (式 11.3)

## 阈值起始范围(附录B,不是结果,须由标定确定)
- 经验成功下界 pW: 0.80–0.90(门内更高)
- 硬净空: > 0 且含标定余量
- 门 commit yaw: 1–3°;门 commit 横向: 0.01–0.04 m
- Weighted A* ε: 1.2–2.5;全局搜索预算: 0.5–2 s;局部反馈: 20–50 Hz
