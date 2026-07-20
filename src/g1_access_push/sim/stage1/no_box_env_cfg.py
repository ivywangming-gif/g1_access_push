"""Stage-1 no-box Unitree G1 environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from agile.rl_env import mdp
from agile.rl_env.assets.robots.unitree_g1 import (
    G1_W_HANDS_AGILE_ACTION_SCALE,
    G1_W_HANDS_AGILE_CFG,
    LEFT_HAND_ARM_JOINT_NAMES,
    LEG_JOINT_NAMES,
    NO_HAND_JOINT_NAMES,
    RIGHT_HAND_ARM_JOINT_NAMES,
    WAIST_JOINT_NAMES,
)
from agile.rl_env.mdp.actions.actions_cfg import (
    AgileLowerBodyActionCfg,
    DeltaJointPositionActionCfg,
)
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils import configclass

AGILE_ROOT = Path(
    os.environ.get(
        "AGILE_PATH",
        "/root/autodl-tmp/robotics/third_party/WBC-AGILE",
    )
)

TEACHER_POLICY_PATH = (
    AGILE_ROOT / "agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher.pt"
)

LEFT_ARM_JOINT_NAMES = LEFT_HAND_ARM_JOINT_NAMES[:-1]
RIGHT_ARM_JOINT_NAMES = RIGHT_HAND_ARM_JOINT_NAMES[:-1]
UPPER_BODY_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES + WAIST_JOINT_NAMES


@configclass
class Stage1SceneCfg(InteractiveSceneCfg):
    """Flat-ground scene containing only one complete G1 robot."""

    terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    robot = G1_W_HANDS_AGILE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Conservative symmetric initial arm posture.
    robot.init_state.pos = (0.0, 0.0, 0.78)
    robot.init_state.joint_pos.update(
        {
            "left_shoulder_pitch_joint": 0.20,
            "left_shoulder_roll_joint": 0.20,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.60,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.20,
            "right_shoulder_roll_joint": -0.20,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.60,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )

    hand_frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=("{ENV_REGEX_NS}/Robot/left_hand/left_hand_palm_link"),
                name="left_hand_palm",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=("{ENV_REGEX_NS}/Robot/right_hand/right_hand_palm_link"),
                name="right_hand_palm",
            ),
        ],
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=2500.0,
        ),
    )


@configclass
class ActionsCfg:
    """Upper-body commands plus frozen AGILE lower-body policy."""

    upper_body_joint_pos = DeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=UPPER_BODY_JOINT_NAMES,
        steady_joint_names=[],
        scale=0.02,
        preserve_order=True,
        joint_limits={
            "waist_roll_joint": (-0.10, 0.10),
            "waist_pitch_joint": (-0.10, 0.10),
            "waist_yaw_joint": (-0.20, 0.20),
        },
    )

    lower_body_joint_pos = AgileLowerBodyActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        obs_group_name="agile_policy",
        policy_path=str(TEACHER_POLICY_PATH),
        policy_output_scale=G1_W_HANDS_AGILE_ACTION_SCALE,
        clip={
            "vx": (-0.20, 0.50),
            "vy": (-0.40, 0.40),
            "wz": (-0.50, 0.50),
            "height": (0.65, 0.72),
        },
    )


@configclass
class ObservationsCfg:
    """Observations for logging and the frozen AGILE teacher."""

    @configclass
    class ControlCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class AgilePolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=NO_HAND_JOINT_NAMES,
                )
            },
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=NO_HAND_JOINT_NAMES,
                )
            },
        )

        actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "lower_body_joint_pos"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    control: ControlCfg = ControlCfg()
    agile_policy: AgilePolicyCfg = AgilePolicyCfg()


@configclass
class G1Stage1NoBoxEnvCfg(ManagerBasedEnvCfg):
    """No-box Stage-1 environment for WBC regression."""

    scene: Stage1SceneCfg = Stage1SceneCfg(
        num_envs=1,
        env_spacing=2.5,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self) -> None:
        self.seed = 42

        # AGILE teacher runs at 50 Hz; physics runs at 200 Hz.
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.hand_frames.update_period = self.decimation * self.sim.dt
