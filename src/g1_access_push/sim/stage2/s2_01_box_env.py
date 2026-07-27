"""IsaacLab scene for S2-01; imports are intentionally runtime-only."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from g1_access_push.sim.stage1.no_box_env_cfg import Stage1SceneCfg
from g1_access_push.sim.stage1.recurrent_no_box_env_cfg import G1Stage1NoBoxRecurrentEnvCfg


S2_01_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.0,
    friction_combine_mode="average",
    restitution_combine_mode="average",
)


@configclass
class S201SceneCfg(Stage1SceneCfg):
    """Certified Stage-1 scene plus one isolated dynamic box and audit sensors."""

    terrain = Stage1SceneCfg.terrain.replace(physics_material=S2_01_MATERIAL)

    box = RigidObjectCfg(
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(25.0, 0.0, 0.602)),
    )

    box_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Box/geometry/mesh",
        filter_prim_paths_expr=["/World/ground/.*"],
        history_length=1,
        track_air_time=False,
    )
    box_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Box/geometry/mesh",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"],
        history_length=1,
        track_air_time=False,
    )
    audit_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AuditCamera",
        update_period=0.0,
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
class G1S201BoxStandEnvCfg(G1Stage1NoBoxRecurrentEnvCfg):
    """One-environment, no-reset, default-arm S2-01 control contract."""

    scene: S201SceneCfg = S201SceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.box_ground_contact.update_period = self.sim.dt
        self.scene.box_robot_contact.update_period = self.sim.dt
        self.scene.audit_camera.update_period = self.decimation * self.sim.dt
