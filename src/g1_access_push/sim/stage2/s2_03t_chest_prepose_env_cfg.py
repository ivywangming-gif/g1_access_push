"""One-environment, no-box configuration for S2-03T chest prepose only."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from agile.rl_env.assets.robots.unitree_g1 import (
    G1_W_HANDS_AGILE_ACTION_SCALE,
    HAND_JOINT_NAMES,
    LEG_JOINT_NAMES,
    WAIST_JOINT_NAMES,
)
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from g1_access_push.sim.stage1.recurrent_no_box_env_cfg import (
    G1Stage1NoBoxRecurrentEnvCfg,
    RecurrentStudentObservationsCfg,
    RecurrentStudentLowerBodyActionCfg,
)
from g1_access_push.sim.stage1.no_box_env_cfg import Stage1SceneCfg
from g1_access_push.stage2.s2_03t_chest_prepose_contract import (
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)


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
            focus_distance=3.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
    )


@configclass
class ChestPreposeSceneCfg(Stage1SceneCfg):
    """The verified no-box G1 scene plus two fixed evidence cameras."""

    chest_prepose_front_camera = _camera(
        "{ENV_REGEX_NS}/ChestPreposeFrontCamera"
    )
    chest_prepose_side_camera = _camera(
        "{ENV_REGEX_NS}/ChestPreposeSideCamera"
    )


@configclass
class ChestPreposeActionsCfg:
    """S2-02 Cartesian arm terms, fixed waist/fingers, frozen legs."""

    left_hand_pose = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(LEFT_ARM_JOINT_NAMES),
        body_name="left_hand_palm_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
        scale=1.0,
    )
    right_hand_pose = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(RIGHT_ARM_JOINT_NAMES),
        body_name="right_hand_palm_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
        scale=1.0,
    )
    waist_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=WAIST_JOINT_NAMES,
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )
    finger_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=HAND_JOINT_NAMES,
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )
    lower_body_joint_pos = RecurrentStudentLowerBodyActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        obs_group_name="student_policy",
        policy_path=(
            "/root/autodl-tmp/robotics/third_party/WBC-AGILE/agile/data/"
            "policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt"
        ),
        policy_output_scale=G1_W_HANDS_AGILE_ACTION_SCALE,
        command_limits={
            "vx": (-0.20, 0.50),
            "vy": (-0.40, 0.40),
            "wz": (-0.50, 0.50),
            "height": (0.65, 0.72),
        },
    )


@configclass
class S203TChestPreposeEnvCfg(G1Stage1NoBoxRecurrentEnvCfg):
    """No-box deterministic chest-prepose environment."""

    scene: ChestPreposeSceneCfg = ChestPreposeSceneCfg(
        num_envs=1,
        env_spacing=2.5,
    )
    observations: RecurrentStudentObservationsCfg = RecurrentStudentObservationsCfg()
    actions: ChestPreposeActionsCfg = ChestPreposeActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scene.num_envs != 1:
            raise ValueError("S2-03T chest prepose requires exactly one environment")
        self.scene.chest_prepose_front_camera.update_period = (
            self.decimation * self.sim.dt
        )
        self.scene.chest_prepose_side_camera.update_period = (
            self.decimation * self.sim.dt
        )
