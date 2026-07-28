"""Runtime-only IsaacLab scene factory for S2-02."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg

from g1_access_push.sim.stage1.virtual_box_env_cfg import G1Stage1VirtualBoxRecurrentEnvCfg
from g1_access_push.sim.stage2.s2_01_box_env import S2_01_MATERIAL
from g1_access_push.stage2.s2_01_process import build_scene_config_instance


BOX_SENSOR_PRIM_PATH = "{ENV_REGEX_NS}/Box"
ROBOT_FILTER_EXPRESSIONS = ("{ENV_REGEX_NS}/Robot/.*",)


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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.90, 0.0, 0.602)),
    )


def build_s2_02_env_cfg() -> G1Stage1VirtualBoxRecurrentEnvCfg:
    """Use the proven S1-07 differential IK and certified recurrent lower body."""
    env_cfg = G1Stage1VirtualBoxRecurrentEnvCfg()
    env_cfg.sim.enable_scene_query_support = True
    base_scene = env_cfg.scene
    scene_cfg = build_scene_config_instance(
        base_scene,
        terrain_cfg=base_scene.terrain.replace(physics_material=S2_01_MATERIAL),
        additions={
            "box": _box_cfg(),
            "box_robot_contact": ContactSensorCfg(
                prim_path=BOX_SENSOR_PRIM_PATH,
                filter_prim_paths_expr=list(ROBOT_FILTER_EXPRESSIONS),
                history_length=1,
                track_air_time=False,
            ),
            "audit_camera": CameraCfg(
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
            ),
        },
    )
    scene_cfg.num_envs = 1
    scene_cfg.box_robot_contact.update_period = env_cfg.sim.dt
    scene_cfg.audit_camera.update_period = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.scene = scene_cfg
    return env_cfg
