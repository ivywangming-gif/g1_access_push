"""Two-view camera configuration used only by S2-03T visual evidence runs."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from g1_access_push.sim.stage2.s2_03t_env_cfg import S203TContactEnvCfg, S203TSceneCfg


VISUAL_WIDTH = 640
VISUAL_HEIGHT = 480


def _camera(prim_path: str) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.02,
        height=VISUAL_HEIGHT,
        width=VISUAL_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
    )


@configclass
class S203TVisualEvidenceSceneCfg(S203TSceneCfg):
    audit_camera = _camera("{ENV_REGEX_NS}/ThreeQuarterFrontCamera")
    side_camera = _camera("{ENV_REGEX_NS}/SideViewCamera")


@configclass
class S203TVisualEvidenceEnvCfg(S203TContactEnvCfg):
    scene: S203TVisualEvidenceSceneCfg = S203TVisualEvidenceSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        update_period = self.decimation * self.sim.dt
        self.scene.audit_camera.update_period = update_period
        self.scene.side_camera.update_period = update_period
