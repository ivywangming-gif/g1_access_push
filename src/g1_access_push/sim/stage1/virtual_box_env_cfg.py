"""Recurrent Stage-1 environment with independent Cartesian arm IK terms."""

from __future__ import annotations

from agile.rl_env.assets.robots.unitree_g1 import (
    G1_W_HANDS_AGILE_ACTION_SCALE,
    LEG_JOINT_NAMES,
    WAIST_JOINT_NAMES,
)
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.utils import configclass

from .no_box_env_cfg import (
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)
from .recurrent_no_box_env_cfg import (
    RECURRENT_STUDENT_POLICY_PATH,
    G1Stage1NoBoxRecurrentEnvCfg,
)
from .recurrent_student_action import (
    RecurrentStudentLowerBodyActionCfg,
)


@configclass
class VirtualBoxActionsCfg:
    """Two Cartesian arm targets, a held waist, and recurrent legs."""

    left_hand_pose = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=LEFT_ARM_JOINT_NAMES,
        body_name="left_hand_palm_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=1.0,
    )

    right_hand_pose = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        body_name="right_hand_palm_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
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

    lower_body_joint_pos = RecurrentStudentLowerBodyActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        obs_group_name="student_policy",
        policy_path=str(RECURRENT_STUDENT_POLICY_PATH),
        policy_output_scale=G1_W_HANDS_AGILE_ACTION_SCALE,
        command_limits={
            "vx": (-0.20, 0.50),
            "vy": (-0.40, 0.40),
            "wz": (-0.50, 0.50),
            "height": (0.65, 0.72),
        },
    )


@configclass
class G1Stage1VirtualBoxRecurrentEnvCfg(G1Stage1NoBoxRecurrentEnvCfg):
    """No physical box; both palms are commanded in the pelvis frame."""

    actions: VirtualBoxActionsCfg = VirtualBoxActionsCfg()
