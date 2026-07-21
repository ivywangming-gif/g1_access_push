"""Stage-1 no-box environment using the recurrent G1 locomotion student."""

from __future__ import annotations

import os
from pathlib import Path

from agile.rl_env import mdp
from agile.rl_env.assets.robots.unitree_g1 import (
    G1_W_HANDS_AGILE_ACTION_SCALE,
    LEG_JOINT_NAMES,
    NO_HAND_JOINT_NAMES,
)
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .no_box_env_cfg import (
    ActionsCfg,
    G1Stage1NoBoxEnvCfg,
    ObservationsCfg,
)
from .recurrent_student_action import (
    RecurrentStudentLowerBodyActionCfg,
)


AGILE_ROOT = Path(
    os.environ.get(
        "AGILE_PATH",
        "/root/autodl-tmp/robotics/third_party/WBC-AGILE",
    )
)

RECURRENT_STUDENT_POLICY_PATH = (
    AGILE_ROOT
    / "agile/data/policy/velocity_height_g1/"
    / "unitree_g1_velocity_height_recurrent_student.pt"
)

# Stage-1 acceptance is deterministic. Actuator-delay robustness
# is evaluated separately instead of being hidden in WBC regression.
STAGE1_RECURRENT_ACTUATOR_DELAY_STEPS = 0
STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES = (
    "legs",
    "feet",
)


@configclass
class RecurrentStudentActionsCfg(ActionsCfg):
    """Existing upper-body action plus recurrent lower-body policy."""

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
class RecurrentStudentObservationsCfg:
    """Logging observations plus the 64-value student state."""

    control: ObservationsCfg.ControlCfg = (
        ObservationsCfg.ControlCfg()
    )

    @configclass
    class StudentPolicyCfg(ObsGroup):
        # Official student order after the four command values:
        # 3 angular velocity
        # 3 projected gravity
        # 29 relative joint positions
        # 29 relative joint velocities
        # Previous 12 policy actions are maintained by the ActionTerm.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity
        )

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

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    student_policy: StudentPolicyCfg = StudentPolicyCfg()


@configclass
class G1Stage1NoBoxRecurrentEnvCfg(
    G1Stage1NoBoxEnvCfg
):
    """Preserve the verified scene while selecting the recurrent policy."""

    observations: RecurrentStudentObservationsCfg = (
        RecurrentStudentObservationsCfg()
    )
    actions: RecurrentStudentActionsCfg = (
        RecurrentStudentActionsCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        for actuator_name in (
            STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES
        ):
            actuator_cfg = self.scene.robot.actuators[
                actuator_name
            ]
            actuator_cfg.min_delay = (
                STAGE1_RECURRENT_ACTUATOR_DELAY_STEPS
            )
            actuator_cfg.max_delay = (
                STAGE1_RECURRENT_ACTUATOR_DELAY_STEPS
            )

        if self.scene.num_envs != 1:
            raise ValueError(
                "Recurrent student integration requires num_envs=1."
            )
