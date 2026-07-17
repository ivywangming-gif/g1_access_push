# 直接在项目根目录运行:python tests/_smoke_logger.py
from g1_access_push.logging.episode_logger import EpisodeLog

log = EpisodeLog(seed=42,
                 geometry={"box": {"L": 1.0, "W": 0.4}, "door": {"D": 0.9}},
                 physics={"mass": 3.0, "mu_ground": 0.4},
                 wbc_id="agile_velocity_g1_history_v1",
                 planner_version="stage0")
log.template_sequence.append("rear_center")
log.add_edge_check(edge="push_0.2m", passed=True, t_ms=1.3)
log.add_state(t=0.0, q_o=[0, 0, 0], chi_L=1, chi_R=1)
log.failure_code = 0
path = log.save(out_dir="experiment_logs")
print("saved:", path)

reloaded = EpisodeLog.load(path)
assert reloaded.seed == 42 and reloaded.wbc_id == log.wbc_id
print("replay OK")