"""Regression tests for the pinned WBC-AGILE actuator-delay contract."""

from g1_access_push.sim.stage1.no_box_env_cfg import (
    G1Stage1NoBoxEnvCfg,
)
from g1_access_push.sim.stage1.recurrent_no_box_env_cfg import (
    STAGE1_RECURRENT_ACTUATOR_DELAY_STEPS,
    STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES,
    G1Stage1NoBoxRecurrentEnvCfg,
)


def _delay_bounds(
    cfg,
    actuator_name: str,
) -> tuple[int, int]:
    actuator_cfg = cfg.scene.robot.actuators[actuator_name]

    return (
        actuator_cfg.min_delay,
        actuator_cfg.max_delay,
    )


def test_recurrent_stage1_uses_pinned_upstream_zero_delay() -> None:
    """Ordinary and recurrent environments both inherit upstream lag zero."""

    ordinary_before = G1Stage1NoBoxEnvCfg()
    recurrent = G1Stage1NoBoxRecurrentEnvCfg()
    ordinary_after = G1Stage1NoBoxEnvCfg()

    assert STAGE1_RECURRENT_ACTUATOR_DELAY_STEPS == 0
    assert STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES == (
        "legs",
        "feet",
    )

    for actuator_name in STAGE1_RECURRENT_DELAYED_ACTUATOR_NAMES:
        assert _delay_bounds(
            ordinary_before,
            actuator_name,
        ) == (0, 0)

        assert _delay_bounds(
            recurrent,
            actuator_name,
        ) == (0, 0)

        assert _delay_bounds(
            ordinary_after,
            actuator_name,
        ) == (0, 0)
