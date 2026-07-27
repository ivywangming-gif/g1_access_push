"""Machine-readable S2-00 result schema; no runtime or simulator imports."""

from __future__ import annotations

from typing import Any

METRIC_FIELDS = (
    "left_contact", "right_contact", "bilateral_contact_duration_s",
    "object_position_xyz_m", "object_translation_from_attach_start_m", "object_yaw_rad",
    "object_yaw_change_rad", "object_linear_speed_mps", "object_yaw_rate_radps",
    "left_contact_force_n", "right_contact_force_n", "left_contact_impulse_ns",
    "right_contact_impulse_ns", "contact_force_peak_n", "contact_force_rate_nps",
    "root_height_m", "root_roll_rad", "root_pitch_rad", "root_tilt_rad",
    "joint_limit_margin_rad", "torque_ratio_max", "foot_contact_state",
    "forbidden_contact_links", "nonfinite", "time_out", "fall", "bad_tilt",
)


def contract_only_result(
    *,
    run_id: str,
    config_path: str,
    config_sha256: str,
    resolved_config_path: str,
    code_head: str,
    branch: str,
) -> dict[str, Any]:
    """Return the required NOT_RUN result without fabricating trace or image evidence."""
    unresolved_metrics = {key: "UNRESOLVED" for key in METRIC_FIELDS}
    return {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "S2-00",
        "task": "attach_light_box_contract",
        "status": "NOT_RUN",
        "primary_reason": "S2_00_STATIC_CONTRACT_ONLY",
        "all_failure_reasons": [],
        "config_path": config_path,
        "resolved_config_path": resolved_config_path,
        "config_sha256": config_sha256,
        "code_head": code_head,
        "branch": branch,
        "source_stage1_head": "fdc908c8ff4cf7179318d53aff37d3ab98b8b797",
        "checkpoint": "NOT_USED",
        "checkpoint_qualification": "NOT_APPLICABLE",
        "episode_count": 0,
        "episode_id": "NOT_RUN",
        "reset_count": 0,
        "auto_reset_detected": False,
        "expected_frames": "UNRESOLVED",
        "observed_frames": 0,
        "trace_path": "NOT_CREATED",
        "final_image_path": "NOT_CREATED",
        "final_image_phase": "NOT_APPLICABLE",
        "timestamps": {"started_at_utc": "NOT_RUN", "finished_at_utc": "NOT_RUN"},
        "fsm_summary": {"states": [], "terminal_state": "NOT_RUN"},
        "metrics": unresolved_metrics,
        "evidence": {"ATTACH_NOT_RUN": True, "BOX_NOT_PUSHED": True},
        "process_audit": {"isaac_started": False, "multiple_isaac_processes": False},
        "contract_flags": {"runnable": False, "qualification_state": "CONTRACT_ONLY_NOT_EXECUTED"},
    }
