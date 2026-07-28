"""Camera-enabled one-environment config used only for evaluation evidence."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from g1_access_push.sim.stage2.s2_03t_env_cfg import S203TContactEnvCfg, S203TSceneCfg


@configclass
class S203TEvaluationSceneCfg(S203TSceneCfg):
    audit_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AuditCamera",
        update_period=0.02,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
    )


@configclass
class S203TContactEvaluationEnvCfg(S203TContactEnvCfg):
    scene: S203TEvaluationSceneCfg = S203TEvaluationSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.audit_camera.update_period = self.decimation * self.sim.dt
