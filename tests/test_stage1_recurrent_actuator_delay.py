"""Regression tests for deterministic recurrent Stage-1 actuator delay."""

from g1_access_push.sim.stage1.no_box_env_cfg import (
    G1Stage1NoBoxEnvCfg,
)
from g1_access_push.sim.stage1.recurrent_no_box_env_cfg import (
    G1Stage1NoBoxRecurrentEnvCfg,
    STAGE1_RECURRENT_ACTUATOR_DELAY_STEPS,
    STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES,
)


def _delay_bounds(cfg, actuator_name: str) -> tuple[int, int]:
    actuator_cfg = cfg.scene.robot.actuators[actuator_name]

    return (
        actuator_cfg.min_delay,
        actuator_cfg.max_delay,
    )


def test_recurrent_stage1_fixes_lower_actuator_delay_to_zero() -> None:
    base_before = G1Stage1NoBoxEnvCfg()

    for actuator_name in (
        STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES
    ):
        assert _delay_bounds(
            base_before,
            actuator_name,
        ) == (0, 4)

    recurrent = G1Stage1NoBoxRecurrentEnvCfg()

    assert STAGE1_RECURRENT_ACTUATOR_DELAY_STEPS == 0
    assert STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES == (
        "legs",
        "feet",
    )

    for actuator_name in (
        STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES
    ):
        assert _delay_bounds(
            recurrent,
            actuator_name,
        ) == (0, 0)

    base_after = G1Stage1NoBoxEnvCfg()

    for actuator_name in (
        STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES
    ):
        assert _delay_bounds(
            base_after,
            actuator_name,
        ) == (0, 4)
