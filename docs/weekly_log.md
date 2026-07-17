# 每周研究日志(§10.15)

## Week 1–2 (Stage 0:冻结问题与接口)
1. 本周冻结了哪个接口/假设?
   - SE(2) 坐标链、箱/门几何与投影、success/failure(F1-F8)、
     WBC adapter(式10.3/10.4)、episode logger、指标(§11.5)、3 个反例场景。
2. 哪个最小测试从失败变为通过?
   - test_frames / test_door_projection / test_success_failure 共 7 项全绿。
3. 失败属于哪类?
   - 无,阶段0 为纯几何/逻辑,无仿真。
4. 新增复杂模块是否有对照证明必要?
   - 未新增 planner/primitive 模块,严格遵守纵向切片。
5. 下周(Stage 1)的一个 go/no-go 指标?
   - 无箱 base+双手跟踪:100 次无跌倒,手/base 误差稳定(§5.10)。
