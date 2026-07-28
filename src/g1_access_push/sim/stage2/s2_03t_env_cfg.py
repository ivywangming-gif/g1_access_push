"""Manager-based RL configuration for S2-03T."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from agile.rl_env import mdp as agile_mdp
from agile.rl_env.assets.robots.unitree_g1 import (
    G1_W_HANDS_AGILE_ACTION_SCALE,
    LEG_JOINT_NAMES,
    NO_HAND_JOINT_NAMES,
)
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from g1_access_push.sim.stage1.no_box_env_cfg import Stage1SceneCfg
from g1_access_push.sim.stage2 import s2_03t_mdp as mdp
from g1_access_push.sim.stage2.s2_01_box_env import S2_01_MATERIAL
from g1_access_push.sim.stage2.s2_03_attach_env import (
    BOX_FILTER_EXPRESSIONS,
    LEFT_PALM_SENSOR_PRIM_PATH,
    RIGHT_PALM_SENSOR_PRIM_PATH,
)
from g1_access_push.sim.stage2.s2_03t_actions import (
    ARM_JOINT_NAMES,
    FrozenRecurrentLowerBodyActionCfg,
    HybridNormalApproachActionCfg,
)
from g1_access_push.stage2.s2_03t_contact_filter_contract import FORBIDDEN_FILTER_EXPRESSIONS

FORBIDDEN_SENSOR_PRIM_PATH = "{ENV_REGEX_NS}/Box"

CERTIFIED_STUDENT = Path(
    "/root/autodl-tmp/robotics/third_party/WBC-AGILE/agile/data/policy/"
    "velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt"
)
CERTIFIED_STUDENT_SHA256 = "a0151975a757a33f0f5ed236d5616e2a98643abe2b36e50ac9ad01b6dd1f6d7e"


def _box_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.6, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=S2_01_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.45, 0.18)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.06, 0.0, 0.602)),
    )


@configclass
class S203TSceneCfg(Stage1SceneCfg):
    """Exact S2 light-box scene without cameras."""

    box = _box_cfg()
    left_palm_box_contact = ContactSensorCfg(
        prim_path=LEFT_PALM_SENSOR_PRIM_PATH,
        filter_prim_paths_expr=list(BOX_FILTER_EXPRESSIONS),
        history_length=2,
        track_air_time=True,
    )
    right_palm_box_contact = ContactSensorCfg(
        prim_path=RIGHT_PALM_SENSOR_PRIM_PATH,
        filter_prim_paths_expr=list(BOX_FILTER_EXPRESSIONS),
        history_length=2,
        track_air_time=True,
    )
    robot_box_contact = ContactSensorCfg(
        prim_path=FORBIDDEN_SENSOR_PRIM_PATH,
        filter_prim_paths_expr=list(FORBIDDEN_FILTER_EXPRESSIONS),
        history_length=2,
        track_air_time=False,
    )


@configclass
class ActionsCfg:
    arm_residual = HybridNormalApproachActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        joint_limit_margin_rad=0.10,
    )
    frozen_lower_body = FrozenRecurrentLowerBodyActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        obs_group_name="student_policy",
        policy_path=str(CERTIFIED_STUDENT),
        expected_sha256=CERTIFIED_STUDENT_SHA256,
        policy_output_scale=G1_W_HANDS_AGILE_ACTION_SCALE,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actor_history = ObsTerm(
            func=mdp.actor_frame_observation,
            history_length=3,
            flatten_history_dim=True,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        actor_history = ObsTerm(
            func=mdp.actor_frame_observation,
            history_length=3,
            flatten_history_dim=True,
        )
        privileged = ObsTerm(func=mdp.critic_privileged_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentPolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=agile_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=agile_mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=agile_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=NO_HAND_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=agile_mdp.joint_vel_rel,
            scale=0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=NO_HAND_JOINT_NAMES)},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    student_policy: StudentPolicyCfg = StudentPolicyCfg()


@configclass
class RewardsCfg:
    gap_progress = RewTerm(func=mdp.gap_progress, weight=250.0)
    approach_symmetry = RewTerm(func=mdp.approach_symmetry, weight=50.0)
    left_contact_onset = RewTerm(func=mdp.left_contact_onset, weight=1.0)
    right_contact_onset = RewTerm(func=mdp.right_contact_onset, weight=1.0)
    bilateral_contact_onset = RewTerm(func=mdp.bilateral_contact_onset, weight=3.0)
    verify_progress = RewTerm(func=mdp.verify_progress, weight=4.0)
    safe_force_band = RewTerm(func=mdp.safe_force_band, weight=1.0)
    single_contact_step = RewTerm(func=mdp.single_contact_step, weight=-0.5)
    contact_retention = RewTerm(func=mdp.contact_retention, weight=1.0)
    attached_hold_step = RewTerm(func=mdp.attached_hold_step, weight=4.0)
    force_balance = RewTerm(func=mdp.force_balance, weight=1.0)
    success_terminal = RewTerm(func=mdp.success_terminal, weight=100.0)
    time_cost = RewTerm(func=mdp.time_cost, weight=-0.01)
    hand_orientation_tracking = RewTerm(func=mdp.hand_orientation_tracking, weight=-0.05)
    unsafe_force = RewTerm(func=mdp.unsafe_force, weight=-4.0)
    force_impulse_rate = RewTerm(func=mdp.force_impulse_rate, weight=-2.0)
    force_imbalance = RewTerm(func=mdp.force_imbalance, weight=-0.5)
    box_translation = RewTerm(func=mdp.box_translation, weight=-40.0)
    box_yaw_change = RewTerm(func=mdp.box_yaw_change, weight=-40.0)
    root_risk = RewTerm(func=mdp.root_risk, weight=-1.0)
    joint_limit_margin = RewTerm(func=mdp.joint_limit_margin, weight=-5.0)
    torque_ratio = RewTerm(func=mdp.torque_ratio, weight=-2.0)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)
    action_rate = RewTerm(func=mdp.action_rate, weight=-0.01)
    hard_force_terminal = RewTerm(func=mdp.hard_force_terminal, weight=-100.0)
    pushing_terminal = RewTerm(func=mdp.pushing_terminal, weight=-100.0)
    contact_loss_terminal = RewTerm(func=mdp.contact_loss_terminal, weight=-25.0)
    forbidden_collision_terminal = RewTerm(func=mdp.forbidden_collision_terminal, weight=-100.0)
    contact_timeout_terminal = RewTerm(func=mdp.contact_timeout_terminal, weight=-25.0)


def _done(metric_name: str) -> DoneTerm:
    return DoneTerm(func=mdp.metric_termination, params={"metric_name": metric_name})


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.success)
    nonfinite = _done("nonfinite")
    forbidden_non_palm_box_collision = _done("forbidden_non_palm_box_collision")
    contact_loss = _done("contact_loss")
    single_hand_timeout = _done("single_hand_timeout")
    force_peak = _done("force_peak")
    palm_impulse = _done("palm_impulse")
    combined_impulse = _done("combined_impulse")
    force_rate = _done("force_rate")
    box_linear_speed = _done("box_linear_speed")
    box_angular_speed = _done("box_angular_speed")
    box_translation = _done("box_translation")
    box_yaw_change = _done("box_yaw_change")
    base_excursion = _done("base_excursion")
    root_height = _done("root_height")
    root_tilt = _done("root_tilt")
    arm_joint_margin = _done("arm_joint_margin")
    arm_torque = _done("arm_torque")
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """No randomization; exact precontact restoration is handled by the env subclass."""

    reset_reference: EventTerm | None = None


@configclass
class CurriculumCfg:
    contact_gap = CurrTerm(func=mdp.contact_gap_curriculum)


@configclass
class S203TContactEnvCfg(ManagerBasedRLEnvCfg):
    scene: S203TSceneCfg = S203TSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum: CurriculumCfg = CurriculumCfg()
    episode_length_s: float = 20.0
    is_finite_horizon: bool = False

    def __post_init__(self) -> None:
        self.seed = 42
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.enable_scene_query_support = True
        self.scene.terrain.physics_material = S2_01_MATERIAL
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.hand_frames.update_period = self.decimation * self.sim.dt
        self.scene.left_palm_box_contact.update_period = self.sim.dt
        self.scene.right_palm_box_contact.update_period = self.sim.dt
        self.scene.robot_box_contact.update_period = self.sim.dt
        if self.episode_length_s / (self.decimation * self.sim.dt) != 1000:
            raise RuntimeError("S2-03T episode must be exactly 1000 control steps")
        for actuator_name in ("legs", "feet"):
            actuator = self.scene.robot.actuators[actuator_name]
            if (actuator.min_delay, actuator.max_delay) != (0, 0):
                raise RuntimeError(f"certified lower-body actuator delay changed: {actuator_name}")
